/**
 * Conduit Tagged Sockets Module
 * Collection and management of tagged sockets across the graph
 */

import { app } from "/scripts/app.js";
import { PROPERTY_KEY } from './core.js';
import { logMinimal } from './logging.js';

/**
 * Canonical Tagged Socket Structure - SINGLE SOURCE OF TRUTH
 * Used everywhere: frontend, backend, logs, external APIs
 *
 * {
 *   // === Identity (unique key) ===
 *   nodeId: number,
 *   slotIndex: number,
 *   side: "input" | "output",
 *
 *   // === Tag info ===
 *   tagName: string,
 *
 *   // === Socket info ===
 *   dataType: string,      // e.g., "STRING", "IMAGE", "CLIP"
 *   slotName: string,      // e.g., "text", "image", "samples"
 *
 *   // === Node info ===
 *   nodeType: string,      // e.g., "CLIPTextEncode", "KSampler"
 *   nodeTitle: string,     // User-set title or defaults to nodeType
 *
 *   // === Connection info ===
 *   linkId: number | null, // Current connection
 * }
 */

/**
 * Build canonical tagged socket object from node and entry data
 */
export function buildTaggedSocket(node, entry, slot) {
    const socket = {
        // Identity
        nodeId: node.id,
        slotIndex: entry.index,
        side: entry.side,
        // Tag info
        tagName: entry.name || `${entry.side}_${node.id}_${entry.index}`,
        // Socket info
        dataType: slot?.type || "*",
        slotName: slot?.name || slot?.label || "unknown",
        // Node info
        nodeType: node.type,
        nodeTitle: node.title || node.type,
        // Connection info (inputs have .link, outputs have .links array)
        linkId: entry.side === "input" ? (slot?.link ?? null) : (slot?.links?.[0] ?? null),
    };

    // Include active status (explicit false means disabled, absence means active)
    if (entry.active === false) {
        socket.active = false;
    }

    return socket;
}

/**
 * Collect all tagged sockets (both inputs and outputs) across the graph.
 * Returns array of canonical tagged socket objects.
 *
 * @param {Object} options - Collection options
 * @param {boolean} options.includeDisabled - If true, include disabled sockets (default: false)
 */
export function collectTaggedSockets({ includeDisabled = false } = {}) {
    const tagged = [];
    const graph = app.graph;
    if (!graph) {
        return tagged;
    }

    for (const node of graph._nodes || []) {
        const data = node.properties?.[PROPERTY_KEY];
        if (!Array.isArray(data)) continue;

        for (const entry of data) {
            // Skip disabled sockets unless includeDisabled is true
            if (!includeDisabled && entry.active === false) continue;

            const slot = entry.side === "input"
                ? node.inputs?.[entry.index]
                : node.outputs?.[entry.index];

            tagged.push(buildTaggedSocket(node, entry, slot));
        }
    }

    return tagged;
}

/**
 * Collect all tagged OUTPUT sockets across the graph.
 * Returns array of canonical tagged socket objects.
 */
export function collectTaggedOutputs() {
    return collectTaggedSockets().filter(s => s.side === "output");
}

/**
 * Collect all tagged INPUT sockets across the graph.
 * Returns array of canonical tagged socket objects.
 */
export function collectTaggedInputs() {
    return collectTaggedSockets().filter(s => s.side === "input");
}

/**
 * Inject ConduitOutput nodes into a prompt object.
 * Modifies the prompt in-place.
 *
 * @param {Object} prompt - The prompt object (node_id -> node_data)
 * @param {string} promptId - Unique ID for this execution
 * @returns {number} Number of nodes injected
 */
export function injectConduitOutputs(prompt, promptId) {
    const tagged = collectTaggedOutputs();
    if (tagged.length === 0) return 0;

    const injectedNodes = [];

    for (const tag of tagged) {
        const injectedId = `conduit_${tag.nodeId}_${tag.slotIndex}`;
        const safeType = tag.dataType.replace(/[^a-zA-Z0-9]/g, '_');

        prompt[injectedId] = {
            class_type: `ConduitOutput_${safeType}`,
            inputs: {
                data: [String(tag.nodeId), tag.slotIndex],
                tag_name: tag.tagName,
                prompt_id: promptId,
            },
        };

        injectedNodes.push({ tagName: tag.tagName, nodeType: tag.nodeType, slotIndex: tag.slotIndex });
    }

    logMinimal("outputs_injected", { count: tagged.length, nodes: injectedNodes });
    return tagged.length;
}
