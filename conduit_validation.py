"""
Conduit Validation - Frontend-parity input validation.

This module validates workflow inputs against the Node Registry before execution.
All validation issues are warnings - execution is never blocked.

Key behaviors:
- INT/FLOAT out of range: Clamp to range + WARNING
- COMBO not in options: WARNING + suggestions, pass raw value
- Unknown node/input: WARNING (drift detection), attempt execution
- Never blocks execution - let ComfyUI handle actual errors
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .conduit_registry import (
    ensure_registry_loaded,
    get_node_def,
    get_input_spec,
    InputKind,
    InputSpec,
    NodeDef,
)


# =============================================================================
# Data Structures
# =============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation issues (no ERROR - we never block)"""
    INFO = "info"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single validation issue"""
    severity: ValidationSeverity
    code: str                    # Machine-readable code
    message: str                 # Human-readable message
    tag_name: Optional[str] = None
    input_name: Optional[str] = None
    node_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "tag_name": self.tag_name,
            "input_name": self.input_name,
            "node_id": self.node_id,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of validating inputs against registry"""
    # Note: No "valid" field - we never block, always attempt execution
    issues: List[ValidationIssue] = field(default_factory=list)

    # Transformations applied
    clamped_values: Dict[str, dict] = field(default_factory=dict)  # tag -> {original, clamped, min, max}

    # Drift detection
    unknown_nodes: List[str] = field(default_factory=list)
    unknown_inputs: List[str] = field(default_factory=list)

    # Frontend logic warnings
    nodes_with_frontend_logic: List[str] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)

    def to_dict(self) -> dict:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "clamped_values": self.clamped_values,
            "unknown_nodes": self.unknown_nodes,
            "unknown_inputs": self.unknown_inputs,
            "nodes_with_frontend_logic": self.nodes_with_frontend_logic,
        }


# =============================================================================
# Main Validation Function
# =============================================================================

async def validate_workflow_inputs(
    workflow: dict,
    input_sockets: List[dict],
    user_inputs: Dict[str, Any]
) -> Tuple[Dict[str, Any], ValidationResult]:
    """
    Validate and potentially transform user inputs against the node registry.

    This function:
    - Validates inputs against object_info constraints
    - Clamps numeric values to min/max ranges
    - Warns on invalid COMBO values with suggestions
    - Detects drift (unknown nodes/inputs)
    - Never blocks execution - all issues are warnings

    Args:
        workflow: The ComfyUI workflow dict
        input_sockets: List of tagged input socket definitions from conduit schema
        user_inputs: Dict of tag_name -> value from the API request

    Returns:
        (transformed_inputs, ValidationResult)

        transformed_inputs has the same keys as user_inputs but with
        values clamped to valid ranges where applicable.
    """
    result = ValidationResult()
    transformed = dict(user_inputs)

    # Ensure registry is loaded
    await ensure_registry_loaded()

    # Track which nodes are in the workflow
    workflow_nodes = set(workflow.keys())

    for socket in input_sockets:
        tag_name = socket.get("tagName", "")
        node_id = str(socket.get("nodeId", ""))
        slot_name = socket.get("slotName", "")
        data_type = socket.get("dataType", "")

        # Skip if input not provided
        if tag_name not in user_inputs:
            continue

        value = user_inputs[tag_name]

        # Skip None values (will use workflow defaults)
        if value is None:
            continue

        # Check if node exists in workflow
        if node_id not in workflow_nodes:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="node_not_in_workflow",
                message=f"Node {node_id} not found in workflow",
                tag_name=tag_name,
                node_id=node_id,
            ))
            continue

        # Get node class type from workflow
        node_data = workflow.get(node_id, {})
        class_type = node_data.get("class_type")

        if not class_type:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="no_class_type",
                message=f"Node {node_id} has no class_type",
                tag_name=tag_name,
                node_id=node_id,
            ))
            continue

        # Get node definition from registry
        node_def = get_node_def(class_type)

        if not node_def:
            if class_type not in result.unknown_nodes:
                result.unknown_nodes.append(class_type)
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="unknown_node_type",
                    message=f"Node type '{class_type}' not in registry - attempting execution",
                    tag_name=tag_name,
                    node_id=node_id,
                ))
            continue

        # Check for frontend logic
        if node_def.has_frontend_logic:
            if class_type not in result.nodes_with_frontend_logic:
                result.nodes_with_frontend_logic.append(class_type)
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="has_frontend_logic",
                    message=f"Node '{class_type}' has frontend logic patterns: {node_def.frontend_risk_patterns}",
                    node_id=node_id,
                    details={"patterns": node_def.frontend_risk_patterns},
                ))

        # Get input spec from registry
        input_spec = node_def.inputs.get(slot_name)

        if not input_spec:
            input_key = f"{class_type}.{slot_name}"
            if input_key not in result.unknown_inputs:
                result.unknown_inputs.append(input_key)
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="unknown_input",
                    message=f"Input '{slot_name}' not in registry for '{class_type}' - attempting execution",
                    tag_name=tag_name,
                    input_name=slot_name,
                ))
            continue

        # Validate and transform based on input kind
        if input_spec.kind == InputKind.COMBO:
            transformed_value, issues = _validate_combo(value, input_spec, tag_name)
            for issue in issues:
                result.add_issue(issue)
            if transformed_value is not None:
                transformed[tag_name] = transformed_value

        elif input_spec.kind == InputKind.PRIMITIVE:
            if input_spec.type_name in ("INT", "FLOAT"):
                transformed_value, issues = _validate_numeric(value, input_spec, tag_name)
                for issue in issues:
                    result.add_issue(issue)
                if transformed_value is not None:
                    transformed[tag_name] = transformed_value
                    # Track clamped values
                    if transformed_value != value and isinstance(transformed_value, (int, float)):
                        result.clamped_values[tag_name] = {
                            "original": value,
                            "clamped": transformed_value,
                            "min": input_spec.min_value,
                            "max": input_spec.max_value,
                        }

    return transformed, result


# =============================================================================
# Type-Specific Validation Functions
# =============================================================================

def _validate_combo(
    value: Any,
    spec: InputSpec,
    tag_name: str
) -> Tuple[Any, List[ValidationIssue]]:
    """
    Validate COMBO input - warn if invalid, pass through anyway.

    Returns (value, issues) - value is unchanged, we don't auto-fix COMBOs.
    """
    issues: List[ValidationIssue] = []

    if spec.options is None or len(spec.options) == 0:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            code="combo_no_options",
            message=f"COMBO '{tag_name}' has no options defined in registry",
            tag_name=tag_name,
        ))
        return value, issues

    # Convert to string for comparison (options can be str, int, or float)
    str_value = str(value)
    str_options = [str(opt) for opt in spec.options]

    # Check exact match
    if str_value in str_options:
        return value, issues

    # Check case-insensitive match
    lower_value = str_value.lower()
    for opt in spec.options:
        if str(opt).lower() == lower_value:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="combo_case_mismatch",
                message=f"Value '{value}' has case mismatch with option '{opt}'",
                tag_name=tag_name,
                details={"provided": value, "matched": opt},
            ))
            # Return the correctly-cased option
            return opt, issues

    # Value not in options - warn with suggestions
    suggestions = _find_similar(str_value, spec.options, max_suggestions=5)

    issues.append(ValidationIssue(
        severity=ValidationSeverity.WARNING,
        code="invalid_combo_value",
        message=f"Value '{value}' not in COMBO options. Suggestions: {suggestions}",
        tag_name=tag_name,
        details={
            "value": value,
            "valid_options": spec.options[:10],  # Limit for readability
            "total_options": len(spec.options),
            "suggestions": suggestions,
        },
    ))

    # Return original value - let ComfyUI fail with its own error
    return value, issues


def _validate_numeric(
    value: Any,
    spec: InputSpec,
    tag_name: str
) -> Tuple[Any, List[ValidationIssue]]:
    """
    Validate numeric input - clamp to range if out of bounds.

    Returns (clamped_value, issues).
    """
    issues: List[ValidationIssue] = []

    # Try to parse the value
    try:
        if spec.type_name == "INT":
            if isinstance(value, bool):
                # Booleans are technically ints in Python, treat specially
                num_value = int(value)
            elif isinstance(value, int):
                num_value = value
            elif isinstance(value, float):
                num_value = int(value)
                if value != num_value:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        code="float_truncated",
                        message=f"Float {value} truncated to int {num_value}",
                        tag_name=tag_name,
                        details={"original": value, "truncated": num_value},
                    ))
            elif isinstance(value, str):
                # Try to parse string
                if '.' in value:
                    num_value = int(float(value))
                else:
                    num_value = int(value)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="string_parsed",
                    message=f"String '{value}' parsed to int {num_value}",
                    tag_name=tag_name,
                ))
            else:
                raise ValueError(f"Cannot convert {type(value).__name__} to INT")
        else:  # FLOAT
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                num_value = float(value)
            elif isinstance(value, str):
                num_value = float(value)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="string_parsed",
                    message=f"String '{value}' parsed to float {num_value}",
                    tag_name=tag_name,
                ))
            else:
                raise ValueError(f"Cannot convert {type(value).__name__} to FLOAT")

    except (ValueError, TypeError) as e:
        # Cannot parse - warn but pass through, let ComfyUI fail
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="parse_error",
            message=f"Cannot parse '{value}' as {spec.type_name}: {e}",
            tag_name=tag_name,
            details={"value": value, "target_type": spec.type_name, "error": str(e)},
        ))
        return value, issues

    # Clamp to range
    original = num_value
    clamped = False

    if spec.min_value is not None and num_value < spec.min_value:
        num_value = int(spec.min_value) if spec.type_name == "INT" else spec.min_value
        clamped = True

    if spec.max_value is not None and num_value > spec.max_value:
        num_value = int(spec.max_value) if spec.type_name == "INT" else spec.max_value
        clamped = True

    if clamped:
        issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="value_clamped",
            message=f"Value {original} clamped to {num_value} (range: {spec.min_value}-{spec.max_value})",
            tag_name=tag_name,
            details={
                "original": original,
                "clamped": num_value,
                "min": spec.min_value,
                "max": spec.max_value,
            },
        ))

    return num_value, issues


def _find_similar(
    value: str,
    options: List[Any],
    max_suggestions: int = 5
) -> List[str]:
    """
    Find similar options using simple string matching.

    Uses a scoring system:
    - Prefix match: +100 points
    - Contains: +50 points
    - Character overlap: +10 points per character
    """
    value_lower = value.lower()
    scored = []

    for opt in options:
        opt_str = str(opt)
        opt_lower = opt_str.lower()
        score = 0

        # Prefix match
        if opt_lower.startswith(value_lower):
            score += 100
        elif value_lower.startswith(opt_lower):
            score += 80

        # Contains
        if value_lower in opt_lower:
            score += 50
        elif opt_lower in value_lower:
            score += 40

        # Character overlap
        overlap = len(set(value_lower) & set(opt_lower))
        score += overlap * 10

        if score > 0:
            scored.append((score, opt_str))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])

    return [opt for _, opt in scored[:max_suggestions]]


print("[Conduit Validation] Module loaded")
