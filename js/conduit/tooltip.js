/**
 * Conduit Tooltip Module
 * Tooltip system for tagged sockets - shows tag name on hover
 */

import { app } from "/scripts/app.js";
import { findSocketTag } from './core.js';
import { getSlotAtPoint } from './socket-detection.js';

let conduitTooltipElement = null;
let tooltipHideTimeout = null;

/**
 * Get or create the tooltip element
 */
function getConduitTooltip() {
    if (!conduitTooltipElement) {
        conduitTooltipElement = document.createElement('div');
        conduitTooltipElement.id = 'conduit-tooltip';
        conduitTooltipElement.style.cssText = `
            position: fixed;
            z-index: 10000;
            background: #1a1a1a;
            border: 1px solid #555;
            border-radius: 6px;
            padding: 6px 10px;
            font-family: Arial, sans-serif;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        `;
        document.body.appendChild(conduitTooltipElement);
    }
    return conduitTooltipElement;
}

/**
 * Show tooltip at position for tagged socket
 * @param {number} x - clientX
 * @param {number} y - clientY
 * @param {string} tagName - the tag name
 * @param {string} socketInfo - e.g. "Input: seed"
 */
function showConduitTooltip(x, y, tagName, socketInfo) {
    const tooltip = getConduitTooltip();

    // Two lines: tag name (bold/colored) on top, socket info below
    tooltip.innerHTML = `
        <div style="color: #ff9966; font-weight: bold; margin-bottom: 3px;">${tagName}</div>
        <div style="color: #aaa;">${socketInfo}</div>
    `;

    // Position tooltip near cursor
    tooltip.style.left = `${x + 15}px`;
    tooltip.style.top = `${y + 10}px`;
    tooltip.style.opacity = '1';

    // Suppress native tooltip by adding a class to document
    document.body.classList.add('conduit-tooltip-active');

    // Clear any pending hide
    if (tooltipHideTimeout) {
        clearTimeout(tooltipHideTimeout);
        tooltipHideTimeout = null;
    }
}

/**
 * Hide the custom tooltip
 */
function hideConduitTooltip() {
    if (tooltipHideTimeout) return; // Already scheduled
    tooltipHideTimeout = setTimeout(() => {
        const tooltip = getConduitTooltip();
        tooltip.style.opacity = '0';
        document.body.classList.remove('conduit-tooltip-active');
        tooltipHideTimeout = null;
    }, 100);
}

/**
 * Get tooltip info for a tagged socket
 * Returns { tagName, socketInfo } or null if not tagged
 */
function getTaggedTooltipInfo(nodeId, side, slotIndex) {
    const node = app.graph?.getNodeById(nodeId);
    if (!node) return null;

    const entry = findSocketTag(node, side, slotIndex);
    if (!entry || entry.active === false) return null;

    // Get slot info for context
    const slot = side === "input" ? node.inputs?.[slotIndex] : node.outputs?.[slotIndex];
    const slotName = slot?.label || slot?.name || slot?.type || "unknown";
    const sideLabel = side === "input" ? "Input" : "Output";
    const socketInfo = `${sideLabel}: ${slotName}`;

    return {
        tagName: entry.name,
        socketInfo: socketInfo,
    };
}

/**
 * Setup tooltip hover handlers
 */
export function setupTooltipHandlers() {
    let currentSlotKey = null;

    document.addEventListener('mousemove', (e) => {
        const slotInfo = getSlotAtPoint(e.clientX, e.clientY);

        if (slotInfo) {
            const key = `${slotInfo.nodeId}-${slotInfo.side}-${slotInfo.slotIndex}`;

            // Check if this socket is tagged
            const tooltipInfo = getTaggedTooltipInfo(slotInfo.nodeId, slotInfo.side, slotInfo.slotIndex);

            if (tooltipInfo) {
                currentSlotKey = key;
                showConduitTooltip(e.clientX, e.clientY, tooltipInfo.tagName, tooltipInfo.socketInfo);
                return;
            }
        }

        // Not over a tagged socket - hide tooltip
        if (currentSlotKey) {
            currentSlotKey = null;
            hideConduitTooltip();
        }
    });
}
