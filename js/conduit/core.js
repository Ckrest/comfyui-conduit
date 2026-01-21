/**
 * Conduit Core Module
 * Data persistence and tag management (toggle/rename/find tags)
 */

import { logNormal } from './logging.js';

export const EXTENSION_NAME = "Conduit";
export const PROPERTY_KEY = "conduit_sockets";

/**
 * Data structure: node.properties.conduit_sockets = [
 *   { side: "input"|"output", index: number, name: string },
 *   ...
 * ]
 *
 * - Entry exists = socket is tagged
 * - On untag: delete if name is default, keep if name was customized
 * - On re-tag: restore existing entry if found, else create new
 */

export function getNodeConduitData(node) {
    node.properties = node.properties || {};
    if (!Array.isArray(node.properties[PROPERTY_KEY])) {
        node.properties[PROPERTY_KEY] = [];
    }
    return node.properties[PROPERTY_KEY];
}

export function findSocketTag(node, side, index) {
    const data = getNodeConduitData(node);
    return data.find(e => e.side === side && e.index === index);
}

export function getDefaultName(node, side, index) {
    const slot = side === "input" ? node.inputs?.[index] : node.outputs?.[index];
    if (!slot) return "unknown";

    // Get slot name: prefer label, then name, then type
    const slotName = slot.label || slot.name || slot.type || "unknown";

    // Get node display name: prefer user-set title, fallback to node type
    const nodeDisplayName = node.title || node.type || "Node";

    // Format: "slotName (nodeDisplayName #nodeId)"
    return `${slotName} (${nodeDisplayName} #${node.id})`;
}

/**
 * Check if an entry's name is the default (not customized)
 */
export function isDefaultName(node, entry) {
    const defaultName = getDefaultName(node, entry.side, entry.index);
    return entry.name === defaultName;
}

/**
 * Toggle a socket tag on/off
 * - If currently active: untag (delete if default name, mark inactive if custom)
 * - If not active: tag (restore inactive entry or create new)
 * Returns: { active: true/false, entry: the entry or null }
 */
export function toggleSocketTag(node, side, index) {
    const data = getNodeConduitData(node);
    const existing = findSocketTag(node, side, index);
    const isActive = existing && existing.active !== false;

    if (isActive) {
        // Currently active → untag
        if (isDefaultName(node, existing)) {
            // Default name → delete entirely
            const idx = data.indexOf(existing);
            data.splice(idx, 1);
            logNormal("tag_toggled", { nodeId: node.id, side, index, active: false });
            return { active: false, entry: null };
        } else {
            // Custom name → keep entry but mark inactive
            existing.active = false;
            logNormal("tag_toggled", { nodeId: node.id, side, index, active: false, kept: true });
            return { active: false, entry: existing };
        }
    } else {
        // Not active → activate
        if (existing) {
            // Restore previous inactive entry (has custom name)
            delete existing.active;  // Remove the false flag
            logNormal("tag_toggled", { nodeId: node.id, side, index, active: true, restored: true });
            return { active: true, entry: existing };
        } else {
            // Create new entry
            const entry = {
                side: side,
                index: index,
                name: getDefaultName(node, side, index),
            };
            data.push(entry);
            logNormal("tag_toggled", { nodeId: node.id, side, index, active: true });
            return { active: true, entry: entry };
        }
    }
}

/**
 * Check if a socket is currently tagged (active)
 */
export function isSocketTagged(node, side, index) {
    const entry = findSocketTag(node, side, index);
    return entry && entry.active !== false;
}

/**
 * Rename a socket tag
 */
export function renameSocketTag(node, side, index, newName) {
    const entry = findSocketTag(node, side, index);

    if (entry && entry.active !== false) {
        const defaultName = getDefaultName(node, side, index);
        entry.name = newName.trim() || defaultName;
        logNormal("tag_renamed", { nodeId: node.id, side, index, newName: entry.name });
        return entry.name;
    }

    return null;
}

/**
 * Check if a tag name is already used by another socket in the graph.
 * Used to enforce unique tag names.
 *
 * @param {LGraph} graph - The LiteGraph graph
 * @param {string} name - The tag name to check
 * @param {number} excludeNodeId - Node ID to exclude (the socket being renamed)
 * @param {string} excludeSide - Side to exclude
 * @param {number} excludeIndex - Index to exclude
 * @returns {boolean} True if name is taken by another socket
 */
export function isTagNameTaken(graph, name, excludeNodeId, excludeSide, excludeIndex) {
    if (!graph || !name) return false;

    const normalizedName = name.trim().toLowerCase();

    for (const node of graph._nodes || []) {
        const data = node?.properties?.[PROPERTY_KEY];
        if (!Array.isArray(data)) continue;

        for (const entry of data) {
            // Skip inactive tags
            if (entry.active === false) continue;

            // Skip the socket being renamed
            if (node.id === excludeNodeId &&
                entry.side === excludeSide &&
                entry.index === excludeIndex) {
                continue;
            }

            // Check for match (case-insensitive)
            if (entry.name?.trim().toLowerCase() === normalizedName) {
                return true;
            }
        }
    }

    return false;
}

/**
 * Get all active tag names in the graph.
 * @param {LGraph} graph - The LiteGraph graph
 * @returns {Set<string>} Set of all active tag names
 */
export function getAllTagNames(graph) {
    const names = new Set();

    if (!graph) return names;

    for (const node of graph._nodes || []) {
        const data = node?.properties?.[PROPERTY_KEY];
        if (!Array.isArray(data)) continue;

        for (const entry of data) {
            if (entry.active !== false && entry.name) {
                names.add(entry.name);
            }
        }
    }

    return names;
}
