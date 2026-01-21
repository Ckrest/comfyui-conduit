/**
 * Conduit Click Handlers Module
 * Right-click tagging and wire monitoring
 */

import { app } from "/scripts/app.js";
import { findSocketTag, getDefaultName, toggleSocketTag, isTagNameTaken } from './core.js';
import { getSlotAtPoint } from './socket-detection.js';
import { scheduleStyleUpdate } from './styles.js';
import { buildTaggedSocket } from './tagged-sockets.js';
import { logNormal, logVerbose } from './logging.js';

// Flag to block context menu when we've handled a slot right-click.
// This is needed because dialog overlays can cover the slot before
// the contextmenu event fires, causing getSlotAtPoint() to fail.
let _blockContextMenu = false;

/**
 * Setup click handlers and wire monitoring.
 * Called from setup() hook - canvas is guaranteed to be ready.
 */
export function setupClickHandlers() {
    const canvas = app.canvas;

    // Right-click handler for tagging
    // Shift+right-click = rename, Right-click = toggle
    document.addEventListener('pointerdown', async (e) => {
        // Right-click only
        if (e.button !== 2) return;

        const slotInfo = getSlotAtPoint(e.clientX, e.clientY);
        if (!slotInfo) return;

        // Set flag to block context menu - cleared after event cycle completes
        _blockContextMenu = true;
        setTimeout(() => { _blockContextMenu = false; }, 0);

        // MUST prevent default BEFORE any await - otherwise browser context menu fires
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        const node = app.graph?.getNodeById(slotInfo.nodeId);
        if (!node) return;

        // Shift+right-click = tag (if needed) + rename
        if (e.shiftKey) {
            let entry = findSocketTag(node, slotInfo.side, slotInfo.slotIndex);

            // If not tagged (or inactive), tag it first
            if (!entry || entry.active === false) {
                const result = toggleSocketTag(node, slotInfo.side, slotInfo.slotIndex);
                entry = result.entry;

                // Update visuals immediately
                scheduleStyleUpdate();
                if (window.conduitPanel) {
                    window.conduitPanel.updateTaggedSockets();
                }
            }

            // Now show rename dialog with validation loop
            if (entry) {
                const defaultName = getDefaultName(node, slotInfo.side, slotInfo.slotIndex);
                let currentName = entry.name || defaultName;
                let errorMessage = null;

                while (true) {
                    const message = errorMessage
                        ? `⚠️ ${errorMessage}\n\nEnter new tag name:`
                        : "Enter new tag name:";

                    const newName = await app.extensionManager.dialog.prompt({
                        title: "Rename Socket Tag",
                        message: message,
                        defaultValue: currentName,
                    });

                    // User cancelled
                    if (newName === null) break;

                    const trimmedName = newName.trim() || defaultName;

                    // Check if name is taken by another socket
                    if (isTagNameTaken(app.graph, trimmedName, node.id, slotInfo.side, slotInfo.slotIndex)) {
                        errorMessage = `"${trimmedName}" is already used by another socket`;
                        currentName = trimmedName;
                        continue;
                    }

                    // Valid name - apply it
                    entry.name = trimmedName;
                    logNormal("tag_renamed_click", { nodeId: node.id, newName: entry.name });
                    if (window.conduitPanel) {
                        window.conduitPanel.updateTaggedSockets();
                    }
                    break;
                }
            }
            return;
        }

        // Regular right-click = toggle tag
        const result = toggleSocketTag(node, slotInfo.side, slotInfo.slotIndex);

        // Build canonical structure for logging
        const slot = slotInfo.side === "input"
            ? node.inputs?.[slotInfo.slotIndex]
            : node.outputs?.[slotInfo.slotIndex];
        const socketData = result.entry
            ? buildTaggedSocket(node, result.entry, slot)
            : { nodeId: slotInfo.nodeId, slotIndex: slotInfo.slotIndex, side: slotInfo.side };

        logNormal("tag_toggled_click", { active: result.active, tagName: socketData.tagName || socketData.slotIndex });

        // Update visual indicators
        scheduleStyleUpdate();

        // Update sidebar if visible
        if (window.conduitPanel) {
            window.conduitPanel.updateTaggedSockets();
        }

    }, { capture: true });

    // Block context menu on slots (or when flag is set from pointerdown)
    document.addEventListener('contextmenu', (e) => {
        // Check flag first - handles case where dialog overlay covers the slot
        if (_blockContextMenu) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            return;
        }
        // Also check slot directly for cases where pointerdown didn't fire
        const slotInfo = getSlotAtPoint(e.clientX, e.clientY);
        if (slotInfo) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }
    }, { capture: true });

    // Also block on pointerup (ComfyUI may trigger menu here)
    document.addEventListener('pointerup', (e) => {
        if (e.button !== 2) return;
        const slotInfo = getSlotAtPoint(e.clientX, e.clientY);
        if (slotInfo) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
        }
    }, { capture: true });

    // Monitor connecting_links for left-click wire drags (for reference)
    let lastConnectingLinks = null;
    const checkConnecting = () => {
        const current = canvas.connecting_links;
        if (current && current.length > 0 && !lastConnectingLinks) {
            const link = current[0];
            const node = link.node;
            const slotIndex = link.slot;
            const side = (link.input !== null && link.input !== undefined) ? "input" : "output";
            const slot = side === "input" ? link.input : link.output;

            // Use consistent identity fields
            const wireInfo = {
                nodeId: node?.id,
                slotIndex: slotIndex,
                side: side,
                slotName: slot?.name || slot?.label || "unknown",
                dataType: slot?.type || "*",
                nodeType: node?.type,
            };

            logVerbose("wire_drag", wireInfo);
        }
        lastConnectingLinks = current;
        requestAnimationFrame(checkConnecting);
    };
    requestAnimationFrame(checkConnecting);

    logNormal("click_handlers_installed");
}
