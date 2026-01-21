/**
 * Conduit - Socket Tagging System for ComfyUI
 * Version: 2.2.0 - Unique tag names enforced
 *
 * Right-click on any socket to toggle its tag on/off.
 * Tag data persists with the workflow in node.properties.
 *
 * Entry point - imports all modules and registers the extension.
 */
console.log("[Conduit] Loading v2.2.0 - Unique tag names enforced");

// Track whether initial graph load is complete
// Used to distinguish "loaded from workflow" vs "pasted during editing"
let graphInitialized = false;

import { app } from "/scripts/app.js";

// Core functionality
import { EXTENSION_NAME, PROPERTY_KEY } from './conduit/core.js';
import { initLogging, logMinimal, logNormal } from './conduit/logging.js';

// UI modules
import { scheduleStyleUpdate, updateTaggedSocketStyles } from './conduit/styles.js';
import { setupTooltipHandlers } from './conduit/tooltip.js';
import { setupClickHandlers } from './conduit/click-handlers.js';
import { registerSidebarTab } from './conduit/panel.js';

// Hook modules
import { setupQueueHook } from './conduit/queue-hook.js';
import { setupSaveHook } from './conduit/save-hook.js';

// ============================================================================
// Extension Registration
// ============================================================================

app.registerExtension({
    name: `Comfy.${EXTENSION_NAME}`,

    /**
     * Called after the application is fully set up.
     * Sets up all Conduit subsystems.
     */
    async setup() {
        await initLogging();

        setupClickHandlers();
        setupTooltipHandlers();
        setupQueueHook();
        setupSaveHook();
        registerSidebarTab();

        // Listen for node removal via LiteGraph callback
        const graph = app.graph;
        if (graph) {
            const originalOnNodeRemoved = graph.onNodeRemoved;
            graph.onNodeRemoved = function(node) {
                if (originalOnNodeRemoved) {
                    originalOnNodeRemoved.call(this, node);
                }
                scheduleStyleUpdate();
                if (window.conduitPanel) {
                    window.conduitPanel.updateTaggedSockets();
                }
            };
        }

        updateTaggedSocketStyles();
        logMinimal("extension_ready", { timestamp: Date.now() });
    },

    /**
     * Official ComfyUI hook: Called after a node's constructor runs.
     * Clears tags from pasted/cloned nodes to enforce manual tagging.
     */
    nodeCreated(node) {
        // Check if this node has tagged sockets
        if (node?.properties?.[PROPERTY_KEY]?.length > 0) {
            if (graphInitialized) {
                // Graph already loaded = this is a paste/clone, clear tags
                const tagCount = node.properties[PROPERTY_KEY].length;
                node.properties[PROPERTY_KEY] = [];
                logNormal("tags_cleared_on_paste", {
                    nodeId: node.id,
                    nodeType: node.type,
                    clearedCount: tagCount
                });
            } else {
                // Initial load = preserve tags
                logNormal("node_loaded_with_tags", {
                    nodeId: node.id,
                    nodeType: node.type,
                    tagCount: node.properties[PROPERTY_KEY].length
                });
            }
        }
    },

    /**
     * Official ComfyUI hook: Called after the graph is configured.
     * Fires when switching tabs, opening workflows, undo/redo, etc.
     * Refreshes the sidebar to show tagged sockets for the current workflow.
     */
    afterConfigureGraph() {
        // Mark initial load complete - subsequent nodeCreated calls are pastes
        graphInitialized = true;
        logNormal("graph_configured");
        scheduleStyleUpdate();
        if (window.conduitPanel) {
            window.conduitPanel.updateTaggedSockets();
        }
    },
});
