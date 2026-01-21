"""
Conduit Registry - Node definition caching with frontend-parity validation.

This module fetches /object_info from ComfyUI, parses it into structured
NodeDef objects, and provides lookup APIs for validation and widget queries.

Key features:
- Lazy-loads on first request (doesn't block startup)
- Caches with configurable TTL
- Scans custom_nodes JS files for frontend logic patterns
- Provides widget query API for clients
"""

import asyncio
import aiohttp
import threading
import time
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from aiohttp import web
import server


# =============================================================================
# Configuration
# =============================================================================

REGISTRY_TTL_SECONDS = 3600  # 1 hour cache TTL
COMFY_API_URL = "http://127.0.0.1:8188"
CUSTOM_NODES_DIR = Path(__file__).parent.parent  # custom_nodes/

# JS patterns that indicate frontend-only logic (legacy - kept for backwards compat)
JS_RISK_PATTERNS = [
    (r"widget\.value\s*=", "widget_value_assignment"),
    (r"widget\.callback\s*=", "widget_callback_assignment"),
    (r"\.addWidget\s*\(", "add_widget"),
    (r"\.addDOMWidget\s*\(", "add_dom_widget"),
    (r"hidden\s*=\s*true", "widget_hiding"),
]

# Enhanced pattern detection for mimicry classification
JS_MIMICRY_PATTERNS = {
    # SIMPLE_REPLICABLE - Can auto-replicate in Python
    "value_passthrough": r"serializeValue\s*[=:]\s*\([^)]*\)\s*=>\s*\{?\s*return\s+\w+\.value",
    "string_format": r"serializeValue.*`\$\{.*\}`",  # Template literals
    "toFixed": r"\.toFixed\((\d+)\)",
    "array_map": r"\.map\s*\(",

    # RANDOMIZATION - Uses randomness we can replicate
    "math_random": r"Math\.random\s*\(",
    "crypto_random": r"crypto\.getRandomValues",

    # STATE_DEPENDENT - Reads other widgets or node state
    "reads_other_widget": r"node\.widgets\[|\.widgets\.find",
    "reads_properties": r"node\.properties\[",
    "widget_hidden_set": r"widget\.hidden\s*=",

    # BROWSER_REQUIRED - Needs browser APIs
    "fetch_api": r"fetch\s*\(|api\.fetchApi",
    "file_upload": r"FormData|FileReader|input\.type\s*=\s*['\"]file",
    "canvas_ops": r"canvas\.|ctx\.|getContext\(",
    "clipboard": r"navigator\.clipboard",
    "webcam": r"getUserMedia|mediaDevices",

    # SERIALIZE_VALUE - Has custom serialization
    "has_serialize_value": r"\.serializeValue\s*=",
    "has_before_queued": r"\.beforeQueued\s*=",
    "has_after_queued": r"\.afterQueued\s*=",
}

# Control widget signature - COMBO with these options indicates a control widget
CONTROL_WIDGET_OPTIONS = {"fixed", "randomize", "increment", "decrement"}
CONTROL_WIDGET_NAME_HINTS = ["control_after_generate", "seed_control", "_control"]


# =============================================================================
# Data Structures
# =============================================================================

class InputKind(Enum):
    """Classification of input types"""
    PRIMITIVE = "primitive"      # INT, FLOAT, STRING, BOOLEAN
    COMBO = "combo"              # List of options (dropdown)
    TYPED = "typed"              # MODEL, CONDITIONING, LATENT, IMAGE, etc.
    UNKNOWN = "unknown"


class MimicryCategory(Enum):
    """Classification of frontend logic for mimicry purposes"""
    UNDETECTED = "undetected"           # No frontend logic detected
    SIMPLE_REPLICABLE = "simple"        # Can auto-replicate (string format, toFixed, etc.)
    RANDOMIZATION = "randomization"     # Uses randomness we can replicate
    STATE_DEPENDENT = "state_dependent" # Reads other widgets/node state
    BROWSER_REQUIRED = "browser"        # Needs browser APIs
    COMPLEX_LOGIC = "complex"           # Too complex to auto-detect


@dataclass
class ControlWidgetPair:
    """A control widget paired with its target widget"""
    control_input: str      # Name of control widget (e.g., "control_after_generate")
    target_input: str       # Name of target widget (e.g., "seed")

    def to_dict(self) -> dict:
        return {
            "control_input": self.control_input,
            "target_input": self.target_input,
        }


@dataclass
class FrontendLogicProfile:
    """Classification of a node's frontend logic for mimicry"""
    # Detected capabilities
    has_serialize_value: bool = False
    has_before_queued: bool = False
    has_after_queued: bool = False

    # Pattern classifications (detected from JS)
    patterns_detected: List[str] = field(default_factory=list)

    # Overall category
    category: MimicryCategory = MimicryCategory.UNDETECTED

    # Replicability confidence (0.0 to 1.0)
    replicability: float = 1.0

    # Control widget pairs detected from object_info
    control_widgets: List[ControlWidgetPair] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_serialize_value": self.has_serialize_value,
            "has_before_queued": self.has_before_queued,
            "has_after_queued": self.has_after_queued,
            "patterns_detected": self.patterns_detected,
            "category": self.category.value,
            "replicability": self.replicability,
            "control_widgets": [cw.to_dict() for cw in self.control_widgets],
        }


@dataclass
class InputSpec:
    """Specification for a single node input"""
    name: str
    kind: InputKind
    type_name: str               # "INT", "FLOAT", "STRING", "BOOLEAN", or custom type
    required: bool

    # For PRIMITIVE types (INT, FLOAT)
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None

    # For COMBO types
    options: Optional[List[Union[str, int, float]]] = None

    # For STRING types
    multiline: bool = False

    # Visibility flags
    hidden: bool = False
    advanced: bool = False
    force_input: bool = False    # Socket required even if widget exists

    # Control widget flag - frontend adds randomize/increment/decrement control
    control_after_generate: bool = False

    # Raw widget config from object_info (for extensibility)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        result = {
            "name": self.name,
            "kind": self.kind.value,
            "type_name": self.type_name,
            "required": self.required,
        }

        if self.default is not None:
            result["default"] = self.default
        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        if self.step is not None:
            result["step"] = self.step
        if self.options is not None:
            result["options"] = self.options
        if self.multiline:
            result["multiline"] = True
        if self.hidden:
            result["hidden"] = True
        if self.advanced:
            result["advanced"] = True
        if self.force_input:
            result["force_input"] = True
        if self.control_after_generate:
            result["control_after_generate"] = True

        return result


@dataclass
class NodeDef:
    """Definition of a ComfyUI node from object_info"""
    class_type: str
    display_name: str
    category: str
    description: str
    python_module: str

    # Input specifications by name
    inputs: Dict[str, InputSpec] = field(default_factory=dict)
    input_order: Dict[str, List[str]] = field(default_factory=dict)

    # Output info
    outputs: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    output_node: bool = False

    # Frontend risk flags (legacy)
    has_frontend_logic: bool = False
    frontend_risk_patterns: List[str] = field(default_factory=list)

    # Enhanced mimicry profile
    mimicry_profile: FrontendLogicProfile = field(default_factory=FrontendLogicProfile)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            "class_type": self.class_type,
            "display_name": self.display_name,
            "category": self.category,
            "description": self.description,
            "python_module": self.python_module,
            "inputs": {name: spec.to_dict() for name, spec in self.inputs.items()},
            "input_order": self.input_order,
            "outputs": self.outputs,
            "output_names": self.output_names,
            "output_node": self.output_node,
            "has_frontend_logic": self.has_frontend_logic,
            "frontend_risk_patterns": self.frontend_risk_patterns,
            "mimicry_profile": self.mimicry_profile.to_dict(),
        }


@dataclass
class RegistryStatus:
    """Status of the node registry"""
    loaded: bool = False
    loading: bool = False
    last_refresh: Optional[str] = None
    node_count: int = 0
    nodes_with_frontend_logic: int = 0
    error: Optional[str] = None
    ttl_seconds: int = REGISTRY_TTL_SECONDS

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "last_refresh": self.last_refresh,
            "node_count": self.node_count,
            "nodes_with_frontend_logic": self.nodes_with_frontend_logic,
            "error": self.error,
            "ttl_seconds": self.ttl_seconds,
        }


# =============================================================================
# Global Registry State
# =============================================================================

_registry: Dict[str, NodeDef] = {}
_registry_lock = threading.RLock()
_registry_status = RegistryStatus()
_load_event = asyncio.Event()


# =============================================================================
# Public API
# =============================================================================

def get_registry_status() -> RegistryStatus:
    """Get current registry status"""
    with _registry_lock:
        return RegistryStatus(
            loaded=_registry_status.loaded,
            loading=_registry_status.loading,
            last_refresh=_registry_status.last_refresh,
            node_count=_registry_status.node_count,
            nodes_with_frontend_logic=_registry_status.nodes_with_frontend_logic,
            error=_registry_status.error,
            ttl_seconds=_registry_status.ttl_seconds,
        )


def get_node_def(class_type: str) -> Optional[NodeDef]:
    """Get node definition by class type"""
    with _registry_lock:
        return _registry.get(class_type)


def get_all_node_defs() -> Dict[str, NodeDef]:
    """Get all node definitions (copy)"""
    with _registry_lock:
        return dict(_registry)


def get_input_spec(class_type: str, input_name: str) -> Optional[InputSpec]:
    """Get input specification for a specific node input"""
    node_def = get_node_def(class_type)
    if node_def:
        return node_def.inputs.get(input_name)
    return None


async def ensure_registry_loaded(force: bool = False) -> bool:
    """
    Ensure registry is loaded. Lazy-loads on first call.

    Args:
        force: If True, force refresh even if cache is valid

    Returns:
        True if registry is available
    """
    with _registry_lock:
        if not force and _registry_status.loaded and not _is_cache_expired():
            return True
        if _registry_status.loading:
            # Wait for existing load to complete
            pass
        else:
            _registry_status.loading = True

    try:
        success = await _fetch_and_parse_object_info()
        if success:
            _scan_custom_nodes_js()
        return success
    finally:
        with _registry_lock:
            _registry_status.loading = False


async def refresh_registry() -> bool:
    """Force refresh the registry"""
    return await ensure_registry_loaded(force=True)


# =============================================================================
# Internal Functions
# =============================================================================

def _is_cache_expired() -> bool:
    """Check if cache is expired"""
    if not _registry_status.last_refresh:
        return True
    try:
        last = datetime.fromisoformat(_registry_status.last_refresh)
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed > REGISTRY_TTL_SECONDS
    except:
        return True


async def _fetch_and_parse_object_info() -> bool:
    """Fetch /object_info and parse into registry"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{COMFY_API_URL}/object_info",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    with _registry_lock:
                        _registry_status.error = f"Failed to fetch object_info: {resp.status}"
                    print(f"[Conduit Registry] Failed to fetch object_info: {resp.status}")
                    return False

                data = await resp.json()

        # Parse into NodeDefs
        with _registry_lock:
            _registry.clear()
            for class_type, info in data.items():
                try:
                    node_def = _parse_node_info(class_type, info)
                    _registry[class_type] = node_def
                except Exception as e:
                    print(f"[Conduit Registry] Error parsing {class_type}: {e}")

            _registry_status.loaded = True
            _registry_status.last_refresh = datetime.now().isoformat()
            _registry_status.node_count = len(_registry)
            _registry_status.error = None

        print(f"[Conduit Registry] Loaded {len(_registry)} nodes from object_info")
        return True

    except Exception as e:
        with _registry_lock:
            _registry_status.error = str(e)
        print(f"[Conduit Registry] Error loading object_info: {e}")
        return False


def _parse_node_info(class_type: str, info: dict) -> NodeDef:
    """Parse object_info entry into NodeDef"""
    node_def = NodeDef(
        class_type=class_type,
        display_name=info.get("display_name", class_type),
        category=info.get("category", ""),
        description=info.get("description", ""),
        python_module=info.get("python_module", ""),
        outputs=list(info.get("output", [])),
        output_names=list(info.get("output_name", [])),
        output_node=info.get("output_node", False),
        input_order=info.get("input_order", {}),
    )

    # Parse inputs from required, optional, and hidden sections
    input_types = info.get("input", {})
    for section in ["required", "optional", "hidden"]:
        section_inputs = input_types.get(section, {})
        for input_name, input_config in section_inputs.items():
            try:
                input_spec = _parse_input_config(
                    input_name,
                    input_config,
                    required=(section == "required"),
                    hidden=(section == "hidden")
                )
                node_def.inputs[input_name] = input_spec
            except Exception as e:
                print(f"[Conduit Registry] Error parsing input {class_type}.{input_name}: {e}")

    return node_def


def _parse_input_config(
    name: str,
    config: Any,
    required: bool,
    hidden: bool = False
) -> InputSpec:
    """Parse a single input configuration"""
    # Config can be: tuple like ("INT", {...}) or ("TYPE",) or (["opt1", "opt2"], {...})

    if not isinstance(config, (list, tuple)) or len(config) == 0:
        return InputSpec(
            name=name,
            kind=InputKind.UNKNOWN,
            type_name="UNKNOWN",
            required=required,
            hidden=hidden
        )

    type_info = config[0]
    widget_config = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}

    # Common options
    default = widget_config.get("default")
    is_hidden = hidden or widget_config.get("hidden", False)
    is_advanced = widget_config.get("advanced", False)
    force_input = widget_config.get("forceInput", False)
    control_after_generate = widget_config.get("control_after_generate", False)

    # COMBO type: first element is a list of options (legacy format)
    if isinstance(type_info, (list, tuple)):
        return InputSpec(
            name=name,
            kind=InputKind.COMBO,
            type_name="COMBO",
            required=required,
            default=default,
            options=list(type_info),
            hidden=is_hidden,
            advanced=is_advanced,
            force_input=force_input,
            raw_config=widget_config,
        )

    # String type name
    if isinstance(type_info, str):
        type_name = type_info.upper()

        # INT with constraints
        if type_name == "INT":
            return InputSpec(
                name=name,
                kind=InputKind.PRIMITIVE,
                type_name="INT",
                required=required,
                default=default,
                min_value=widget_config.get("min"),
                max_value=widget_config.get("max"),
                step=widget_config.get("step"),
                hidden=is_hidden,
                advanced=is_advanced,
                force_input=force_input,
                control_after_generate=control_after_generate,
                raw_config=widget_config,
            )

        # FLOAT with constraints
        if type_name == "FLOAT":
            return InputSpec(
                name=name,
                kind=InputKind.PRIMITIVE,
                type_name="FLOAT",
                required=required,
                default=default,
                min_value=widget_config.get("min"),
                max_value=widget_config.get("max"),
                step=widget_config.get("step"),
                hidden=is_hidden,
                advanced=is_advanced,
                force_input=force_input,
                control_after_generate=control_after_generate,
                raw_config=widget_config,
            )

        # STRING
        if type_name == "STRING":
            return InputSpec(
                name=name,
                kind=InputKind.PRIMITIVE,
                type_name="STRING",
                required=required,
                default=default,
                multiline=widget_config.get("multiline", False),
                hidden=is_hidden,
                advanced=is_advanced,
                force_input=force_input,
                raw_config=widget_config,
            )

        # BOOLEAN
        if type_name == "BOOLEAN":
            return InputSpec(
                name=name,
                kind=InputKind.PRIMITIVE,
                type_name="BOOLEAN",
                required=required,
                default=default,
                hidden=is_hidden,
                advanced=is_advanced,
                force_input=force_input,
                raw_config=widget_config,
            )

        # COMBO (V2 format - type is "COMBO" with options in config)
        if type_name == "COMBO":
            return InputSpec(
                name=name,
                kind=InputKind.COMBO,
                type_name="COMBO",
                required=required,
                default=default,
                options=widget_config.get("options", []),
                hidden=is_hidden,
                advanced=is_advanced,
                force_input=force_input,
                raw_config=widget_config,
            )

        # Other typed inputs (MODEL, CONDITIONING, IMAGE, LATENT, CLIP, VAE, etc.)
        return InputSpec(
            name=name,
            kind=InputKind.TYPED,
            type_name=type_name,
            required=required,
            default=default,
            hidden=is_hidden,
            advanced=is_advanced,
            force_input=force_input,
            raw_config=widget_config,
        )

    return InputSpec(
        name=name,
        kind=InputKind.UNKNOWN,
        type_name="UNKNOWN",
        required=required,
        hidden=hidden
    )


def _detect_control_widgets(node_def: NodeDef) -> List[ControlWidgetPair]:
    """
    Detect control widget pairs in a node based on input signatures.

    A control widget is a COMBO with options like ["fixed", "randomize", "increment", "decrement"]
    that affects another widget (usually a seed/INT widget).
    """
    pairs = []

    for input_name, spec in node_def.inputs.items():
        # Look for COMBO inputs with control-like options
        if spec.kind != InputKind.COMBO or not spec.options:
            continue

        options_set = set(str(o).lower() for o in spec.options)

        # Check for control widget signature
        if not CONTROL_WIDGET_OPTIONS.issubset(options_set):
            continue

        # Found a control widget! Now find its target.
        target = _find_control_target(input_name, node_def)
        if target:
            pairs.append(ControlWidgetPair(
                control_input=input_name,
                target_input=target
            ))

    return pairs


def _find_control_target(control_name: str, node_def: NodeDef) -> Optional[str]:
    """
    Infer which widget a control widget affects.

    Heuristics:
    1. Name matching: "seed_control" → "seed", "control_after_generate" paired with "seed"
    2. Position: control follows target in input order
    3. Type matching: control near INT input (for seed-like widgets)
    """
    # Heuristic 1: Name-based matching
    for suffix in CONTROL_WIDGET_NAME_HINTS:
        if suffix in control_name.lower():
            # Try to extract base name
            base = control_name.lower().replace(suffix, "").strip("_")
            # Look for matching input
            for name in node_def.inputs:
                if name.lower() == base or base in name.lower():
                    return name

    # Heuristic 2: Standard "control_after_generate" pairs with "seed"
    if "control" in control_name.lower():
        if "seed" in node_def.inputs:
            return "seed"
        if "noise_seed" in node_def.inputs:
            return "noise_seed"

    # Heuristic 3: Look for adjacent INT widget in input order
    input_order = (
        node_def.input_order.get("required", []) +
        node_def.input_order.get("optional", [])
    )
    try:
        idx = input_order.index(control_name)
        if idx > 0:
            prev_input = input_order[idx - 1]
            prev_spec = node_def.inputs.get(prev_input)
            if prev_spec and prev_spec.type_name == "INT":
                return prev_input
    except (ValueError, IndexError):
        pass

    return None


def _classify_mimicry_category(patterns: List[str]) -> tuple[MimicryCategory, float]:
    """
    Classify patterns into a mimicry category with confidence score.

    Returns (category, replicability) where replicability is 0.0 to 1.0.
    """
    if not patterns:
        return MimicryCategory.UNDETECTED, 1.0

    # Check for browser-required patterns (lowest replicability)
    browser_patterns = {"fetch_api", "file_upload", "canvas_ops", "clipboard", "webcam"}
    if browser_patterns.intersection(patterns):
        return MimicryCategory.BROWSER_REQUIRED, 0.1

    # Check for state-dependent patterns
    state_patterns = {"reads_other_widget", "reads_properties", "widget_hidden_set"}
    if state_patterns.intersection(patterns):
        return MimicryCategory.STATE_DEPENDENT, 0.7

    # Check for randomization patterns
    random_patterns = {"math_random", "crypto_random"}
    if random_patterns.intersection(patterns):
        return MimicryCategory.RANDOMIZATION, 1.0

    # Check for simple replicable patterns
    simple_patterns = {"value_passthrough", "string_format", "toFixed", "array_map"}
    if simple_patterns.intersection(patterns):
        return MimicryCategory.SIMPLE_REPLICABLE, 0.95

    # Has serialize/queued hooks but unknown patterns
    hook_patterns = {"has_serialize_value", "has_before_queued", "has_after_queued"}
    if hook_patterns.intersection(patterns):
        return MimicryCategory.COMPLEX_LOGIC, 0.5

    return MimicryCategory.UNDETECTED, 1.0


def _scan_custom_nodes_js():
    """
    Scan custom_nodes JS files for frontend logic patterns.

    This populates both legacy frontend_risk_patterns and enhanced mimicry_profile
    for each node in the registry.
    """
    print("[Conduit Registry] Scanning custom_nodes for frontend logic patterns...")

    # Map: folder name -> (legacy_patterns, mimicry_patterns)
    folder_patterns: Dict[str, tuple[Set[str], Set[str]]] = {}

    for node_folder in CUSTOM_NODES_DIR.iterdir():
        if not node_folder.is_dir() or node_folder.name.startswith((".", "_")):
            continue

        legacy_patterns: Set[str] = set()
        mimicry_patterns: Set[str] = set()

        # Scan all JS files in the folder (including subdirectories)
        for js_file in node_folder.rglob("*.js"):
            try:
                content = js_file.read_text(errors="ignore")

                # Legacy patterns
                for pattern, pattern_name in JS_RISK_PATTERNS:
                    if re.search(pattern, content):
                        legacy_patterns.add(pattern_name)

                # Enhanced mimicry patterns
                for pattern_name, pattern in JS_MIMICRY_PATTERNS.items():
                    if re.search(pattern, content):
                        mimicry_patterns.add(pattern_name)

            except Exception:
                pass

        if legacy_patterns or mimicry_patterns:
            folder_patterns[node_folder.name] = (legacy_patterns, mimicry_patterns)

    # Associate patterns with nodes based on python_module
    flagged_count = 0
    control_widget_count = 0

    with _registry_lock:
        for class_type, node_def in _registry.items():
            # Detect control widgets from object_info (works for all nodes)
            control_widgets = _detect_control_widgets(node_def)
            if control_widgets:
                node_def.mimicry_profile.control_widgets = control_widgets
                control_widget_count += len(control_widgets)

            # Try to match node to custom_node folder based on python_module
            python_module = node_def.python_module or ""
            module_parts = python_module.split(".")

            matched = False
            for folder_name, (legacy, mimicry) in folder_patterns.items():
                # Check if folder name appears in python_module path
                folder_lower = folder_name.lower().replace("-", "_").replace(" ", "_")
                for part in module_parts:
                    part_lower = part.lower().replace("-", "_").replace(" ", "_")
                    if folder_lower == part_lower or folder_lower in part_lower:
                        # Legacy flags
                        node_def.has_frontend_logic = True
                        node_def.frontend_risk_patterns = list(legacy)

                        # Enhanced mimicry profile
                        node_def.mimicry_profile.patterns_detected = list(mimicry)
                        node_def.mimicry_profile.has_serialize_value = "has_serialize_value" in mimicry
                        node_def.mimicry_profile.has_before_queued = "has_before_queued" in mimicry
                        node_def.mimicry_profile.has_after_queued = "has_after_queued" in mimicry

                        # Classify category and replicability
                        category, replicability = _classify_mimicry_category(list(mimicry))
                        node_def.mimicry_profile.category = category
                        node_def.mimicry_profile.replicability = replicability

                        flagged_count += 1
                        matched = True
                        break
                if matched:
                    break

        _registry_status.nodes_with_frontend_logic = flagged_count

    print(f"[Conduit Registry] Flagged {flagged_count} nodes with frontend logic patterns")
    print(f"[Conduit Registry] Detected {control_widget_count} control widget pairs")


# =============================================================================
# API Endpoints
# =============================================================================

@server.PromptServer.instance.routes.get("/conduit/registry/status")
async def registry_status_endpoint(request):
    """Get registry status"""
    status = get_registry_status()
    return web.json_response(status.to_dict())


@server.PromptServer.instance.routes.post("/conduit/registry/refresh")
async def registry_refresh_endpoint(request):
    """Force refresh the registry"""
    success = await refresh_registry()
    status = get_registry_status()
    return web.json_response({
        "success": success,
        "node_count": status.node_count,
        "error": status.error,
    })


@server.PromptServer.instance.routes.get("/conduit/registry/node/{class_type}")
async def registry_node_endpoint(request):
    """Get node definition by class type"""
    class_type = request.match_info["class_type"]

    await ensure_registry_loaded()
    node_def = get_node_def(class_type)

    if not node_def:
        return web.json_response(
            {"status": "error", "message": f"Node not found: {class_type}"},
            status=404
        )

    return web.json_response(node_def.to_dict())


@server.PromptServer.instance.routes.get("/conduit/registry/widget/{class_type}/{input_name}")
async def registry_widget_endpoint(request):
    """
    Get widget info for a specific input.

    Returns the full InputSpec including options for COMBO, min/max for numeric, etc.
    """
    class_type = request.match_info["class_type"]
    input_name = request.match_info["input_name"]

    await ensure_registry_loaded()

    node_def = get_node_def(class_type)
    if not node_def:
        return web.json_response(
            {"status": "error", "message": f"Node not found: {class_type}"},
            status=404
        )

    input_spec = node_def.inputs.get(input_name)
    if not input_spec:
        return web.json_response(
            {"status": "error", "message": f"Input not found: {class_type}.{input_name}"},
            status=404
        )

    result = input_spec.to_dict()
    result["class_type"] = class_type
    result["input_name"] = input_name

    return web.json_response(result)


@server.PromptServer.instance.routes.get("/conduit/registry/widget-options/{class_type}/{input_name}")
async def registry_widget_options_endpoint(request):
    """
    Get possible values for a widget.

    For COMBO: Returns list of valid options
    For INT/FLOAT: Returns {min, max, step, default}
    For STRING: Returns {default, multiline}
    For BOOLEAN: Returns {default}
    """
    class_type = request.match_info["class_type"]
    input_name = request.match_info["input_name"]

    await ensure_registry_loaded()

    input_spec = get_input_spec(class_type, input_name)
    if not input_spec:
        return web.json_response(
            {"status": "error", "message": f"Input not found: {class_type}.{input_name}"},
            status=404
        )

    result = {
        "class_type": class_type,
        "input_name": input_name,
        "kind": input_spec.kind.value,
        "type_name": input_spec.type_name,
    }

    if input_spec.kind == InputKind.COMBO:
        result["options"] = input_spec.options or []
        result["default"] = input_spec.default
    elif input_spec.kind == InputKind.PRIMITIVE:
        if input_spec.type_name in ("INT", "FLOAT"):
            result["min"] = input_spec.min_value
            result["max"] = input_spec.max_value
            result["step"] = input_spec.step
            result["default"] = input_spec.default
        elif input_spec.type_name == "STRING":
            result["default"] = input_spec.default
            result["multiline"] = input_spec.multiline
        elif input_spec.type_name == "BOOLEAN":
            result["default"] = input_spec.default

    return web.json_response(result)


@server.PromptServer.instance.routes.get("/conduit/workflows/{name}/inputs")
async def workflow_inputs_endpoint(request):
    """
    Get all tagged inputs for a workflow, enriched with registry constraints.

    This combines the workflow's tagged socket schema with node definitions
    from the registry to provide complete widget info for each input.

    Response includes:
    - workflow_value: The value saved in the workflow (what you'd get if you leave blank)
    - registry: Node definition data including defaults, min/max, COMBO options

    Query params:
    - refresh: If "true", forces a registry refresh for fresh COMBO options
    - include_disabled: If "true", includes sockets with active: false
    """
    from .conduit_workflows import get_workflow_path

    workflow_name = request.match_info["name"]
    file_path = get_workflow_path(workflow_name)

    if not file_path.exists():
        return web.json_response(
            {"status": "error", "message": f"Workflow not found: {workflow_name}"},
            status=404
        )

    try:
        saved = json.loads(file_path.read_text())
    except Exception as e:
        return web.json_response(
            {"status": "error", "message": f"Failed to load workflow: {e}"},
            status=500
        )

    # Refresh registry for fresh COMBO options (checkpoints, LoRAs, etc.)
    # This ensures dropdown options reflect current filesystem state
    should_refresh = request.query.get("refresh", "").lower() == "true"
    if should_refresh:
        await refresh_registry()
    else:
        await ensure_registry_loaded()

    # Check if we should include disabled sockets
    include_disabled = request.query.get("include_disabled", "").lower() == "true"

    schema = saved.get("schema", {})
    input_sockets = schema.get("inputs", [])
    workflow = saved.get("workflow", {})

    enriched_inputs = []
    for socket in input_sockets:
        # Filter disabled sockets unless include_disabled is true
        # A socket is disabled if it has explicit active: false
        if not include_disabled and socket.get("active") is False:
            continue
        tag_name = socket.get("tagName", "")
        node_id = str(socket.get("nodeId", ""))
        slot_name = socket.get("slotName", "")
        data_type = socket.get("dataType", "")

        # Get class_type and inputs from workflow
        node_data = workflow.get(node_id, {})
        class_type = node_data.get("class_type", "")
        node_inputs = node_data.get("inputs", {})

        # Extract the workflow value for this input
        # This is what's saved in the workflow JSON - what you get if you don't override
        workflow_value = node_inputs.get(slot_name)

        # Handle linked inputs (value is [node_id, output_index])
        # In this case, we can't show a simple value - it's connected to another node
        is_linked = isinstance(workflow_value, list) and len(workflow_value) == 2

        # Build enriched input info
        input_info = {
            "tag_name": tag_name,
            "node_id": node_id,
            "slot_name": slot_name,
            "data_type": data_type,
            "class_type": class_type,
            "workflow_value": None if is_linked else workflow_value,
            "is_linked": is_linked,
        }

        # Add registry data if available
        if class_type:
            node_def = get_node_def(class_type)
            if node_def:
                input_spec = node_def.inputs.get(slot_name)
                if input_spec:
                    input_info["registry"] = input_spec.to_dict()
                    input_info["has_frontend_logic"] = node_def.has_frontend_logic
                else:
                    input_info["registry"] = None
                    input_info["registry_error"] = f"Input '{slot_name}' not found in registry"
            else:
                input_info["registry"] = None
                input_info["registry_error"] = f"Node '{class_type}' not found in registry"
        else:
            input_info["registry"] = None
            input_info["registry_error"] = "No class_type in workflow"

        enriched_inputs.append(input_info)

    return web.json_response({
        "workflow_name": workflow_name,
        "inputs": enriched_inputs,
    })


print("[Conduit Registry] Endpoints registered:")
print("  > GET  /conduit/registry/status")
print("  > POST /conduit/registry/refresh")
print("  > GET  /conduit/registry/node/{class_type}")
print("  > GET  /conduit/registry/widget/{class_type}/{input_name}")
print("  > GET  /conduit/registry/widget-options/{class_type}/{input_name}")
print("  > GET  /conduit/workflows/{name}/inputs")
