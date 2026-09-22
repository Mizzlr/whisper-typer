#!/usr/bin/python3
"""Small system-Python bridge for Ubuntu's pyatspi package."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import pyatspi


@dataclass
class Limits:
    depth: int
    nodes: int
    visited: int = 0


def safe(call, default=None):
    try:
        return call()
    except Exception:
        return default


def applications():
    desktop = pyatspi.Registry.getDesktop(0)
    return [desktop.getChildAtIndex(index) for index in range(desktop.childCount)]


def descendants(root):
    stack = [(root, [])]
    while stack:
        node, path = stack.pop()
        yield node, path
        count = safe(lambda: node.childCount, 0) or 0
        for index in reversed(range(count)):
            child = safe(lambda index=index: node.getChildAtIndex(index))
            if child is not None:
                stack.append((child, path + [index]))


def choose_root(pid: int, title: str):
    apps = applications()
    exact = [app for app in apps if safe(app.get_process_id) == pid]
    candidates = exact or apps
    title_folded = title.casefold().strip()
    if title_folded:
        partial = []
        for app in candidates:
            for node, _ in descendants(app):
                name = safe(lambda: node.name, "") or ""
                if name.casefold() == title_folded:
                    return node
                if name and (name.casefold() in title_folded or title_folded in name.casefold()):
                    partial.append((len(name), node))
        if partial:
            return max(partial, key=lambda item: item[0])[1]
    if len(exact) == 1:
        return exact[0]
    if not exact:
        raise RuntimeError(f"No accessibility application matches PID {pid}")
    raise RuntimeError(f"Multiple accessibility applications match PID {pid}")


def actions(node):
    interface = safe(node.queryAction)
    if not interface:
        return []
    return [safe(lambda index=index: interface.getName(index), "") for index in range(interface.nActions)]


def extents(node):
    component = safe(node.queryComponent)
    rect = safe(lambda: component.getExtents(pyatspi.DESKTOP_COORDS)) if component else None
    if not rect:
        return None
    return {"x": rect.x, "y": rect.y, "width": rect.width, "height": rect.height}


def serialize(node, path: list[int], depth: int, limits: Limits):
    if limits.visited >= limits.nodes:
        return None
    limits.visited += 1
    result = {
        "path": ".".join(map(str, path)) or "root",
        "role": safe(node.getRoleName, "unknown"),
        "name": safe(lambda: node.name, "") or "",
    }
    description = safe(lambda: node.description, "") or ""
    node_actions = actions(node)
    bounds = extents(node)
    if description:
        result["description"] = description
    if node_actions:
        result["actions"] = node_actions
    if bounds and bounds["width"] > 0 and bounds["height"] > 0:
        result["bounds"] = bounds
    if depth < limits.depth:
        children = []
        count = safe(lambda: node.childCount, 0) or 0
        for index in range(count):
            if limits.visited >= limits.nodes:
                break
            child = safe(lambda index=index: node.getChildAtIndex(index))
            if child is not None:
                encoded = serialize(child, path + [index], depth + 1, limits)
                if encoded:
                    children.append(encoded)
        if children:
            result["children"] = children
    return result


def activate(root, name: str, role: str, requested_action: str):
    wanted_name = name.casefold()
    wanted_role = role.casefold().strip()
    matches = []
    for node, path in descendants(root):
        node_name = safe(lambda: node.name, "") or ""
        node_role = safe(node.getRoleName, "") or ""
        if node_name.casefold() == wanted_name and (not wanted_role or node_role.casefold() == wanted_role):
            node_actions = actions(node)
            if node_actions:
                matches.append((node, path, node_role, node_actions))
    if not matches:
        raise RuntimeError(f"No actionable accessible element exactly matches name={name!r}, role={role!r}")
    if len(matches) > 1:
        summary = [{"path": ".".join(map(str, path)), "role": node_role, "actions": node_actions} for _, path, node_role, node_actions in matches]
        raise RuntimeError(f"Accessible target is ambiguous: {json.dumps(summary)}")
    node, path, node_role, node_actions = matches[0]
    selected = requested_action or next((candidate for candidate in ("click", "press", "activate", "jump") if candidate in node_actions), node_actions[0])
    if selected not in node_actions:
        raise RuntimeError(f"Action {selected!r} is unavailable; choices: {node_actions}")
    interface = node.queryAction()
    if not interface.doAction(node_actions.index(selected)):
        raise RuntimeError(f"Accessible action {selected!r} returned failure")
    return {"path": ".".join(map(str, path)) or "root", "name": name, "role": node_role, "action": selected}


def terminal_regions(root):
    regions = []
    for node, path in descendants(root):
        if safe(node.getRoleName, "") != "terminal":
            continue
        bounds = extents(node)
        if bounds and bounds["width"] > 0 and bounds["height"] > 0:
            regions.append({"path": ".".join(map(str, path)) or "root", "bounds": bounds})
    return sorted(regions, key=lambda item: (item["bounds"]["y"], item["bounds"]["x"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "activate", "terminals"))
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--window-title", default="")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--name", default="")
    parser.add_argument("--role", default="")
    parser.add_argument("--action", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        root = choose_root(args.pid, args.window_title)
        if args.mode == "snapshot":
            limits = Limits(args.max_depth, args.max_nodes)
            result = {"application_pid": args.pid, "node_count": 0, "tree": serialize(root, [], 0, limits)}
            result["node_count"] = limits.visited
            result["truncated"] = limits.visited >= limits.nodes
        elif args.mode == "activate":
            result = activate(root, args.name, args.role, args.action)
        else:
            result = terminal_regions(root)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
