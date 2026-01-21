/**
 * Conduit Styles Module
 * Visual indicators for tagged sockets (CSS-based)
 */

import { app } from "/scripts/app.js";
import { PROPERTY_KEY } from './core.js';
import { logVerbose } from './logging.js';

let conduitStyleElement = null;
let styleUpdateTimeout = null;

/**
 * Get or create the style element for Conduit visual indicators
 */
function getConduitStyleElement() {
    if (!conduitStyleElement) {
        conduitStyleElement = document.createElement('style');
        conduitStyleElement.id = 'conduit-socket-styles';
        document.head.appendChild(conduitStyleElement);

        // Add base styles that don't change (tooltip suppression)
        const baseStyles = document.createElement('style');
        baseStyles.id = 'conduit-base-styles';
        baseStyles.textContent = `
            /* Hide native ComfyUI tooltips when our custom tooltip is showing */
            body.conduit-tooltip-active .comfy-tooltip,
            body.conduit-tooltip-active [data-tooltip],
            body.conduit-tooltip-active .p-tooltip {
                display: none !important;
                opacity: 0 !important;
                visibility: hidden !important;
            }
        `;
        document.head.appendChild(baseStyles);
    }
    return conduitStyleElement;
}

/**
 * Generate a slot key from node ID, side, and index
 * Format: "nodeId-in-index" or "nodeId-out-index"
 */
export function makeSlotKey(nodeId, side, index) {
    const inOut = side === "input" ? "in" : "out";
    return `${nodeId}-${inOut}-${index}`;
}

/**
 * Update CSS to show visual indicators on all tagged sockets
 * Input sockets get blue glow, output sockets get red glow
 */
export function updateTaggedSocketStyles() {
    const styleEl = getConduitStyleElement();
    const inputRules = [];
    const outputRules = [];

    // Iterate all nodes in the graph
    const graph = app.graph;
    if (!graph) {
        return;
    }

    for (const node of graph._nodes || []) {
        const data = node.properties?.[PROPERTY_KEY];
        if (!Array.isArray(data)) continue;

        for (const entry of data) {
            // Only style active entries
            if (entry.active === false) continue;

            const key = makeSlotKey(node.id, entry.side, entry.index);
            if (entry.side === "input") {
                inputRules.push(`[data-slot-key="${key}"]`);
            } else {
                outputRules.push(`[data-slot-key="${key}"]`);
            }
        }
    }

    logVerbose("styles_updated", { tagged: inputRules.length + outputRules.length }, { debounce: true });

    const allRules = [...inputRules, ...outputRules];
    if (allRules.length > 0) {
        // Build parent visibility rules - target the opacity-0 div inside lg-node-widget
        const parentRules = allRules.map(r => `.lg-node-widget:has(${r}) > .opacity-0`);

        let css = `
            /* Force the hidden slot container visible when it contains a tagged socket */
            ${parentRules.join(',\n')} {
                opacity: 1 !important;
            }
        `;

        // Input tags - blue glow
        if (inputRules.length > 0) {
            css += `
            /* Input socket tags - blue glow */
            ${inputRules.join(',\n')} {
                box-shadow: 0 0 0 3px rgba(100, 150, 255, 0.8);
                border-radius: 50%;
            }
            `;
        }

        // Output tags - red glow (existing behavior)
        if (outputRules.length > 0) {
            css += `
            /* Output socket tags - red glow */
            ${outputRules.join(',\n')} {
                box-shadow: 0 0 0 3px rgba(255, 100, 100, 0.8);
                border-radius: 50%;
            }
            `;
        }

        styleEl.textContent = css;
    } else {
        styleEl.textContent = '';
    }
}

/**
 * Schedule a style update (debounced)
 */
export function scheduleStyleUpdate() {
    if (styleUpdateTimeout) return;
    styleUpdateTimeout = setTimeout(() => {
        styleUpdateTimeout = null;
        updateTaggedSocketStyles();
    }, 50);
}
