"""Real MCP server tool definitions for benchmarks.

Tool corpus from 6 popular MCP servers (~138 tools total):
- Chrome DevTools MCP (26 tools)
- GitHub MCP (51 tools)
- Playwright MCP (20 tools)
- Filesystem MCP (14 tools)
- Git MCP (12 tools)
- Notion MCP (15 tools)

All definitions in Anthropic ``input_schema`` format, transcribed from
the public source / docs of each server.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Chrome DevTools MCP — 26 tools
# Source: https://github.com/ChromeDevTools/chrome-devtools-mcp
# ---------------------------------------------------------------------------

_CHROME_DEVTOOLS_TOOLS: list[dict] = [
    # -- Navigation / Pages --
    {
        "name": "navigate_page",
        "description": "Navigate to a URL on the currently selected page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "reload_page",
        "description": "Reload the currently selected page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ignoreCache": {
                    "type": "boolean",
                    "description": "If true, bypass the browser cache.",
                },
            },
        },
    },
    {
        "name": "go_back",
        "description": "Navigate to the previous page in the browser history.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "go_forward",
        "description": "Navigate to the next page in the browser history.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_pages",
        "description": "List all open pages/tabs in the connected browser.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "select_page",
        "description": "Select a page/tab by its targetId to interact with.",
        "input_schema": {
            "type": "object",
            "properties": {
                "targetId": {
                    "type": "string",
                    "description": "The targetId of the page to select.",
                },
            },
            "required": ["targetId"],
        },
    },
    {
        "name": "new_page",
        "description": "Open a new blank page/tab in the browser, optionally navigating to a URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Optional URL to navigate to after opening the page.",
                },
            },
        },
    },
    {
        "name": "close_page",
        "description": "Close the currently selected page/tab.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "wait_for",
        "description": "Wait for the specified text to appear on the selected page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to wait for on the page.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Maximum wait time in milliseconds.",
                },
            },
            "required": ["text", "timeout"],
        },
    },
    # -- Input --
    {
        "name": "click",
        "description": "Click on an element on the page identified by its uid from the page snapshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "The uid of the element to click from the page snapshot.",
                },
            },
            "required": ["uid"],
        },
    },
    {
        "name": "hover",
        "description": "Hover over an element on the page identified by its uid from the page snapshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "The uid of the element to hover over.",
                },
            },
            "required": ["uid"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into the currently focused element on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to type."},
                "submit": {
                    "type": "boolean",
                    "description": "Whether to press Enter after typing.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a keyboard key or key combination on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key to press (e.g. 'Enter', 'Tab', 'ArrowDown').",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "select_option",
        "description": "Select an option from a dropdown/select element on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "The uid of the select element.",
                },
                "value": {
                    "type": "string",
                    "description": "The value of the option to select.",
                },
            },
            "required": ["uid", "value"],
        },
    },
    {
        "name": "check_element",
        "description": "Check or uncheck a checkbox element on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "The uid of the checkbox element.",
                },
                "checked": {
                    "type": "boolean",
                    "description": "Whether to check or uncheck the element.",
                },
            },
            "required": ["uid", "checked"],
        },
    },
    {
        "name": "scroll_page",
        "description": "Scroll the page or an element by the given amount.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "The uid of the element to scroll. If omitted, scrolls the page.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "The direction to scroll.",
                },
            },
            "required": ["direction"],
        },
    },
    # -- Snapshot / Screenshot --
    {
        "name": "take_snapshot",
        "description": "Capture a text-based accessibility tree snapshot of the current page, listing elements with unique identifiers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verbose": {
                    "type": "boolean",
                    "description": "Include comprehensive accessibility tree data.",
                },
                "filePath": {
                    "type": "string",
                    "description": "Path to save the snapshot to instead of returning inline.",
                },
            },
        },
    },
    {
        "name": "take_screenshot",
        "description": "Take a screenshot of the page or a specific element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "webp"],
                    "description": "Image format. Default is 'png'.",
                },
                "quality": {
                    "type": "number",
                    "description": "Compression quality 0-100 for JPEG/WebP.",
                },
                "uid": {
                    "type": "string",
                    "description": "The uid of an element to screenshot. If omitted, captures the full page.",
                },
                "fullPage": {
                    "type": "boolean",
                    "description": "If true, capture the full scrollable page.",
                },
                "filePath": {
                    "type": "string",
                    "description": "Path to save the screenshot to.",
                },
            },
        },
    },
    # -- Console --
    {
        "name": "list_console_messages",
        "description": "List all console messages for the currently selected page since the last navigation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pageSize": {
                    "type": "number",
                    "description": "Maximum number of messages to return.",
                },
                "pageIdx": {
                    "type": "number",
                    "description": "Page number to return (0-based).",
                },
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by message types: log, debug, info, error, warn, etc.",
                },
            },
        },
    },
    {
        "name": "get_console_message",
        "description": "Get a specific console message by its id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "msgid": {
                    "type": "number",
                    "description": "The msgid of the console message.",
                },
            },
            "required": ["msgid"],
        },
    },
    # -- Network --
    {
        "name": "list_network_requests",
        "description": "List all network requests for the currently selected page since the last navigation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pageSize": {
                    "type": "number",
                    "description": "Maximum number of requests to return.",
                },
                "pageIdx": {
                    "type": "number",
                    "description": "Page number to return (0-based).",
                },
                "resourceTypes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by resource types: document, script, xhr, fetch, image, etc.",
                },
            },
        },
    },
    {
        "name": "get_network_request",
        "description": "Get a specific network request by its reqid, including headers, body, and response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reqid": {
                    "type": "number",
                    "description": "The reqid of the network request.",
                },
                "requestFilePath": {
                    "type": "string",
                    "description": "Path to save the request body to.",
                },
                "responseFilePath": {
                    "type": "string",
                    "description": "Path to save the response body to.",
                },
            },
        },
    },
    # -- Performance --
    {
        "name": "performance_start_trace",
        "description": "Start a performance trace recording to identify performance issues and deliver Core Web Vital scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reload": {
                    "type": "boolean",
                    "description": "Whether to reload the page after tracing begins.",
                },
                "autoStop": {
                    "type": "boolean",
                    "description": "Whether trace recording stops automatically.",
                },
                "filePath": {
                    "type": "string",
                    "description": "Path to save raw trace data.",
                },
            },
        },
    },
    {
        "name": "performance_stop_trace",
        "description": "Stop the active performance trace recording on the selected page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "Path to save raw trace data.",
                },
            },
        },
    },
    {
        "name": "performance_analyze_insight",
        "description": "Return detailed information on a specific Performance Insight from a prior trace recording.",
        "input_schema": {
            "type": "object",
            "properties": {
                "insightSetId": {
                    "type": "string",
                    "description": "Identifier from available insight sets list.",
                },
                "insightName": {
                    "type": "string",
                    "description": "Insight name to analyze (e.g. 'DocumentLatency', 'LCPBreakdown').",
                },
            },
            "required": ["insightSetId", "insightName"],
        },
    },
    # -- Script / Evaluate --
    {
        "name": "evaluate_script",
        "description": "Evaluate a JavaScript expression or function on the currently selected page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The JavaScript expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
    },
]

# ---------------------------------------------------------------------------
# GitHub MCP — 51 tools
# Source: https://github.com/github/github-mcp-server
# ---------------------------------------------------------------------------

_GITHUB_TOOLS: list[dict] = [
    {
        "name": "get_me",
        "description": "Get details of the authenticated GitHub user.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_issue",
        "description": "Get details of a specific issue in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner."},
                "repo": {"type": "string", "description": "Repository name."},
                "issue_number": {"type": "number", "description": "Issue number."},
            },
            "required": ["owner", "repo", "issue_number"],
        },
    },
    {
        "name": "list_issues",
        "description": "List issues in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner."},
                "repo": {"type": "string", "description": "Repository name."},
                "state": {"type": "string", "description": "Filter by state: open, closed, all."},
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Filter by labels."},
                "page": {"type": "number", "description": "Page number for pagination."},
                "perPage": {"type": "number", "description": "Results per page (max 100)."},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "create_issue",
        "description": "Create a new issue in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner."},
                "repo": {"type": "string", "description": "Repository name."},
                "title": {"type": "string", "description": "Issue title."},
                "body": {"type": "string", "description": "Issue body/description."},
                "assignees": {"type": "array", "items": {"type": "string"}, "description": "Assignees."},
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Labels."},
            },
            "required": ["owner", "repo", "title"],
        },
    },
    {
        "name": "update_issue",
        "description": "Update an existing issue in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner."},
                "repo": {"type": "string", "description": "Repository name."},
                "issue_number": {"type": "number", "description": "Issue number."},
                "title": {"type": "string", "description": "New title."},
                "body": {"type": "string", "description": "New description."},
                "state": {"type": "string", "description": "New state: open, closed."},
                "labels": {"type": "array", "items": {"type": "string"}, "description": "New labels."},
            },
            "required": ["owner", "repo", "issue_number"],
        },
    },
    {
        "name": "add_issue_comment",
        "description": "Add a comment to an existing issue in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner."},
                "repo": {"type": "string", "description": "Repository name."},
                "issue_number": {"type": "number", "description": "Issue number."},
                "body": {"type": "string", "description": "Comment body."},
            },
            "required": ["owner", "repo", "issue_number", "body"],
        },
    },
    {
        "name": "search_issues",
        "description": "Search for issues across GitHub repositories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Search query using GitHub issues search syntax."},
                "sort": {"type": "string", "description": "Sort field."},
                "order": {"type": "string", "description": "Sort order: asc, desc."},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["q"],
        },
    },
    {
        "name": "create_pull_request",
        "description": "Create a new pull request in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner."},
                "repo": {"type": "string", "description": "Repository name."},
                "title": {"type": "string", "description": "PR title."},
                "body": {"type": "string", "description": "PR description."},
                "head": {"type": "string", "description": "Branch with changes."},
                "base": {"type": "string", "description": "Branch to merge into."},
                "draft": {"type": "boolean", "description": "Create as draft PR."},
            },
            "required": ["owner", "repo", "title", "head", "base"],
        },
    },
    {
        "name": "list_pull_requests",
        "description": "List pull requests in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string", "description": "Filter: open, closed, all."},
                "head": {"type": "string", "description": "Filter by head branch."},
                "base": {"type": "string", "description": "Filter by base branch."},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_pull_request",
        "description": "Get details of a specific pull request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number", "description": "Pull request number."},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "update_pull_request",
        "description": "Update an existing pull request in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "state": {"type": "string"},
                "base": {"type": "string"},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "merge_pull_request",
        "description": "Merge a pull request in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
                "merge_method": {"type": "string", "description": "merge, squash, or rebase."},
                "commit_title": {"type": "string"},
                "commit_message": {"type": "string"},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "get_pull_request_diff",
        "description": "Get the diff of a pull request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "get_pull_request_files",
        "description": "Get the list of files changed in a pull request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "get_pull_request_status",
        "description": "Get the combined status and check runs for a pull request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "update_pull_request_branch",
        "description": "Update a pull request branch with the latest changes from the base branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
                "expectedHeadSha": {"type": "string"},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "create_pull_request_review",
        "description": "Create a review on a pull request with comments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
                "event": {"type": "string", "description": "APPROVE, REQUEST_CHANGES, or COMMENT."},
                "body": {"type": "string"},
                "comments": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["owner", "repo", "pullNumber", "event"],
        },
    },
    {
        "name": "submit_pending_pull_request_review",
        "description": "Submit a pending pull request review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
                "event": {"type": "string", "description": "APPROVE, REQUEST_CHANGES, or COMMENT."},
                "body": {"type": "string"},
            },
            "required": ["owner", "repo", "pullNumber", "event"],
        },
    },
    {
        "name": "request_copilot_review",
        "description": "Request a GitHub Copilot code review for a pull request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pullNumber": {"type": "number"},
            },
            "required": ["owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "get_file_contents",
        "description": "Get the contents of a file or directory from a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {"type": "string", "description": "File or directory path."},
                "ref": {"type": "string", "description": "Branch, tag, or commit SHA."},
            },
            "required": ["owner", "repo", "path"],
        },
    },
    {
        "name": "create_or_update_file",
        "description": "Create or update a single file in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "content": {"type": "string"},
                "message": {"type": "string", "description": "Commit message."},
                "branch": {"type": "string"},
                "sha": {"type": "string", "description": "SHA of file being replaced (for updates)."},
            },
            "required": ["owner", "repo", "path", "content", "message", "branch"],
        },
    },
    {
        "name": "push_files",
        "description": "Push multiple files to a GitHub repository in a single commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "branch": {"type": "string"},
                "files": {"type": "array", "items": {"type": "object"}, "description": "Array of {path, content}."},
                "message": {"type": "string", "description": "Commit message."},
            },
            "required": ["owner", "repo", "branch", "files", "message"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for code across GitHub repositories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Search query using GitHub code search syntax."},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["q"],
        },
    },
    {
        "name": "search_repositories",
        "description": "Search for GitHub repositories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_users",
        "description": "Search for GitHub users.",
        "input_schema": {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Search query."},
                "sort": {"type": "string"},
                "order": {"type": "string"},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["q"],
        },
    },
    {
        "name": "get_repository",
        "description": "Get details about a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_repository_tree",
        "description": "Get the file tree of a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "sha": {"type": "string", "description": "Branch, tag, or tree SHA."},
                "recursive": {"type": "boolean"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "create_repository",
        "description": "Create a new GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Repository name."},
                "description": {"type": "string"},
                "private": {"type": "boolean"},
                "autoInit": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "fork_repository",
        "description": "Fork a GitHub repository to your account or an organization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "organization": {"type": "string"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "create_branch",
        "description": "Create a new branch in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "branch": {"type": "string", "description": "New branch name."},
                "from_branch": {"type": "string", "description": "Source branch."},
            },
            "required": ["owner", "repo", "branch"],
        },
    },
    {
        "name": "list_branches",
        "description": "List branches in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "list_commits",
        "description": "Get the list of commits for a branch in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "sha": {"type": "string", "description": "Branch name or SHA."},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "list_tags",
        "description": "List git tags in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_tag",
        "description": "Get details about a specific git tag in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "tag": {"type": "string"},
            },
            "required": ["owner", "repo", "tag"],
        },
    },
    {
        "name": "list_releases",
        "description": "List releases in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "list_notifications",
        "description": "List all GitHub notifications for the authenticated user, including unread mentions, review requests, and assignments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "ISO 8601 timestamp."},
                "filter": {"type": "string", "description": "default, include_read_notifications, or only_participating."},
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
        },
    },
    {
        "name": "mark_all_notifications_read",
        "description": "Mark all GitHub notifications as read.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lastReadAt": {"type": "string", "description": "ISO 8601 timestamp."},
                "owner": {"type": "string"},
                "repo": {"type": "string"},
            },
        },
    },
    {
        "name": "manage_notification_subscription",
        "description": "Manage a notification subscription: ignore, watch, or delete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "notificationID": {"type": "string"},
                "action": {"type": "string", "description": "ignore, watch, or delete."},
            },
            "required": ["notificationID", "action"],
        },
    },
    {
        "name": "manage_repository_notification_subscription",
        "description": "Manage a repository notification subscription: ignore, watch, or delete repository notifications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "action": {"type": "string", "description": "ignore, watch, or delete."},
            },
            "required": ["owner", "repo", "action"],
        },
    },
    {
        "name": "get_code_scanning_alert",
        "description": "Get details of a specific code scanning alert in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "alertNumber": {"type": "number"},
            },
            "required": ["owner", "repo", "alertNumber"],
        },
    },
    {
        "name": "list_code_scanning_alerts",
        "description": "List code scanning alerts in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "severity": {"type": "string", "description": "critical, high, medium, low."},
                "state": {"type": "string", "description": "open, closed, dismissed."},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_secret_scanning_alert",
        "description": "Get details of a specific secret scanning alert in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "alertNumber": {"type": "number"},
            },
            "required": ["owner", "repo", "alertNumber"],
        },
    },
    {
        "name": "list_secret_scanning_alerts",
        "description": "List secret scanning alerts in a GitHub repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string", "description": "open, resolved."},
                "resolution": {"type": "string"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "list_workflow_runs",
        "description": "List workflow runs for a GitHub Actions workflow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "workflowId": {"type": "string"},
                "branch": {"type": "string"},
                "status": {"type": "string"},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_workflow_run",
        "description": "Get details of a specific GitHub Actions workflow run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "runId": {"type": "number"},
            },
            "required": ["owner", "repo", "runId"],
        },
    },
    {
        "name": "cancel_workflow_run",
        "description": "Cancel a running GitHub Actions workflow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "runId": {"type": "number"},
            },
            "required": ["owner", "repo", "runId"],
        },
    },
    {
        "name": "run_workflow",
        "description": "Trigger a GitHub Actions workflow dispatch event.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "workflowId": {"type": "string"},
                "ref": {"type": "string", "description": "Branch or tag."},
                "inputs": {"type": "object", "description": "Workflow input parameters."},
            },
            "required": ["owner", "repo", "workflowId", "ref"],
        },
    },
    {
        "name": "rerun_failed_jobs",
        "description": "Re-run only the failed jobs in a GitHub Actions workflow run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "runId": {"type": "number"},
            },
            "required": ["owner", "repo", "runId"],
        },
    },
    {
        "name": "get_job_logs",
        "description": "Get logs for a specific job in a GitHub Actions workflow run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "jobId": {"type": "number"},
            },
            "required": ["owner", "repo", "jobId"],
        },
    },
    {
        "name": "download_workflow_run_artifact",
        "description": "Download an artifact from a GitHub Actions workflow run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "artifactId": {"type": "number"},
            },
            "required": ["owner", "repo", "artifactId"],
        },
    },
    {
        "name": "get_team_members",
        "description": "Get members of a GitHub team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "org": {"type": "string", "description": "Organization name."},
                "teamSlug": {"type": "string", "description": "Team slug."},
            },
            "required": ["org", "teamSlug"],
        },
    },
    {
        "name": "get_teams",
        "description": "List teams in a GitHub organization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "org": {"type": "string"},
                "page": {"type": "number"},
                "perPage": {"type": "number"},
            },
            "required": ["org"],
        },
    },
]

# ---------------------------------------------------------------------------
# Playwright MCP — 20 tools (core subset)
# Source: https://github.com/microsoft/playwright-mcp
# ---------------------------------------------------------------------------

_PLAYWRIGHT_TOOLS: list[dict] = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "browser_navigate_back",
        "description": "Go back to the previous page in browser history.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_click",
        "description": "Perform a click on a web page element identified by its accessibility ref.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {"type": "string", "description": "Human-readable element description for logging."},
                "ref": {"type": "string", "description": "Exact target element ref from the page snapshot."},
            },
            "required": ["element", "ref"],
        },
    },
    {
        "name": "browser_hover",
        "description": "Hover over an element on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {"type": "string"},
                "ref": {"type": "string"},
            },
            "required": ["element", "ref"],
        },
    },
    {
        "name": "browser_type",
        "description": "Type text into an editable element on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {"type": "string"},
                "ref": {"type": "string"},
                "text": {"type": "string", "description": "Text to type."},
                "submit": {"type": "boolean", "description": "Press Enter after typing."},
            },
            "required": ["element", "ref", "text"],
        },
    },
    {
        "name": "browser_press_key",
        "description": "Press a key or key combination on the keyboard.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name (e.g. 'Enter', 'ArrowDown', 'Control+c')."},
            },
            "required": ["key"],
        },
    },
    {
        "name": "browser_select_option",
        "description": "Select an option in a dropdown by value or label.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element": {"type": "string"},
                "ref": {"type": "string"},
                "value": {"type": "string", "description": "Option value to select."},
            },
            "required": ["element", "ref", "value"],
        },
    },
    {
        "name": "browser_snapshot",
        "description": "Capture an accessibility snapshot of the current page for understanding page structure.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_take_screenshot",
        "description": "Take a screenshot of the current page or a specific element.",
        "input_schema": {
            "type": "object",
            "properties": {
                "raw": {"type": "boolean", "description": "Return raw base64 data without saving."},
            },
        },
    },
    {
        "name": "browser_handle_dialog",
        "description": "Handle a JavaScript dialog (alert, confirm, prompt) that appeared on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "accept": {"type": "boolean", "description": "Accept or dismiss the dialog."},
                "promptText": {"type": "string", "description": "Text to enter in a prompt dialog."},
            },
            "required": ["accept"],
        },
    },
    {
        "name": "browser_file_upload",
        "description": "Upload one or more files to a file input element on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Absolute file paths to upload.",
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "browser_fill_form",
        "description": "Fill multiple form fields at once on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Array of {ref, value} field entries.",
                },
            },
            "required": ["fields"],
        },
    },
    {
        "name": "browser_evaluate",
        "description": "Evaluate a JavaScript expression in the browser page context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "JavaScript expression to evaluate."},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "browser_console_messages",
        "description": "Return all console messages captured from the browser page.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_network_requests",
        "description": "Return all network requests captured since loading the page.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_tabs",
        "description": "List, create, close, or select a browser tab.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "close", "select"],
                    "description": "Tab action to perform.",
                },
                "url": {"type": "string", "description": "URL for creating a new tab."},
                "tabId": {"type": "string", "description": "Tab ID for close/select actions."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "browser_close",
        "description": "Close the current browser page.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "browser_wait_for",
        "description": "Wait for text to appear or disappear on the page, or wait for a specified duration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to wait for."},
                "timeout": {"type": "number", "description": "Maximum wait time in ms."},
                "state": {"type": "string", "enum": ["visible", "hidden"], "description": "Wait for visible or hidden."},
            },
        },
    },
    {
        "name": "browser_resize",
        "description": "Resize the browser window to specified dimensions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "width": {"type": "number", "description": "New width in pixels."},
                "height": {"type": "number", "description": "New height in pixels."},
            },
            "required": ["width", "height"],
        },
    },
    {
        "name": "browser_drag",
        "description": "Perform drag and drop between two elements on the page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "startElement": {"type": "string"},
                "startRef": {"type": "string"},
                "endElement": {"type": "string"},
                "endRef": {"type": "string"},
            },
            "required": ["startElement", "startRef", "endElement", "endRef"],
        },
    },
]

# ---------------------------------------------------------------------------
# Filesystem MCP — 14 tools
# Source: https://github.com/modelcontextprotocol/servers  (filesystem)
# ---------------------------------------------------------------------------

_FILESYSTEM_TOOLS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read the complete contents of a file from the filesystem. Handles text encodings and can read partial content with head/tail parameters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
                "head": {"type": "number", "description": "Read only the first N lines."},
                "tail": {"type": "number", "description": "Read only the last N lines."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_multiple_files",
        "description": "Read multiple files simultaneously. Efficient for batch reading and comparing files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read.",
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "write_file",
        "description": "Create a new file or completely overwrite an existing file with new content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
                "content": {"type": "string", "description": "File content to write."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Make line-based text edits to a file, showing a git-style diff of changes. Supports dry-run mode.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Array of {oldText, newText} edit operations.",
                },
                "dryRun": {"type": "boolean", "description": "Preview changes without applying."},
            },
            "required": ["path", "edits"],
        },
    },
    {
        "name": "create_directory",
        "description": "Create a new directory or ensure a directory exists. Creates parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": "Get a detailed listing of files and subdirectories in a directory path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory_with_sizes",
        "description": "Get a directory listing including file sizes, sortable by name or size.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "sortBy": {"type": "string", "enum": ["name", "size"], "description": "Sort order."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "directory_tree",
        "description": "Get a recursive tree view of files and directories as JSON.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "excludePatterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns to exclude.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "move_file",
        "description": "Move or rename files and directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path."},
                "destination": {"type": "string", "description": "Destination path."},
            },
            "required": ["source", "destination"],
        },
    },
    {
        "name": "search_files",
        "description": "Recursively search for files and directories matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Starting directory."},
                "pattern": {"type": "string", "description": "Glob pattern to match."},
                "excludePatterns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["path", "pattern"],
        },
    },
    {
        "name": "get_file_info",
        "description": "Get detailed metadata about a file or directory including size, timestamps, and permissions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_allowed_directories",
        "description": "Return the list of directories that the server is allowed to access.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_text_file",
        "description": "Read a text file with encoding detection. Supports head and tail parameters for partial reads.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "head": {"type": "number"},
                "tail": {"type": "number"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_media_file",
        "description": "Read an image or audio file. Returns base64 encoded data and MIME type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
]

# ---------------------------------------------------------------------------
# Git MCP — 12 tools
# Source: https://github.com/modelcontextprotocol/servers  (git)
# ---------------------------------------------------------------------------

_GIT_TOOLS: list[dict] = [
    {
        "name": "git_status",
        "description": "Show the working tree status of a git repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Path to git repository."},
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_diff_unstaged",
        "description": "Show unstaged changes in the working directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "context_lines": {"type": "number", "description": "Number of context lines."},
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_diff_staged",
        "description": "Show changes that are staged for the next commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "context_lines": {"type": "number"},
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_diff",
        "description": "Show differences between branches, commits, or other references.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "target": {"type": "string", "description": "Branch or commit to diff against."},
                "context_lines": {"type": "number"},
            },
            "required": ["repo_path", "target"],
        },
    },
    {
        "name": "git_commit",
        "description": "Record staged changes to the repository with a commit message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "message": {"type": "string", "description": "Commit message."},
            },
            "required": ["repo_path", "message"],
        },
    },
    {
        "name": "git_add",
        "description": "Add file contents to the git staging area.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to stage.",
                },
            },
            "required": ["repo_path", "files"],
        },
    },
    {
        "name": "git_reset",
        "description": "Unstage all staged changes in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_log",
        "description": "Show the commit log history of the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "max_count": {"type": "number", "description": "Maximum commits to return."},
                "start_timestamp": {"type": "string", "description": "Filter commits after this date."},
                "end_timestamp": {"type": "string", "description": "Filter commits before this date."},
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_create_branch",
        "description": "Create a new git branch from an optional base branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "branch_name": {"type": "string"},
                "base_branch": {"type": "string"},
            },
            "required": ["repo_path", "branch_name"],
        },
    },
    {
        "name": "git_checkout",
        "description": "Switch to a different branch in the git repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "branch_name": {"type": "string"},
            },
            "required": ["repo_path", "branch_name"],
        },
    },
    {
        "name": "git_show",
        "description": "Show the contents of a specific commit (message, author, diff).",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "revision": {"type": "string", "description": "Commit SHA or reference."},
            },
            "required": ["repo_path", "revision"],
        },
    },
    {
        "name": "git_branch",
        "description": "List branches in the git repository, filtered by local, remote, or all.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "branch_type": {"type": "string", "description": "local, remote, or all."},
                "contains": {"type": "string", "description": "Filter to branches containing this commit."},
            },
            "required": ["repo_path"],
        },
    },
]

# ---------------------------------------------------------------------------
# Notion MCP — 15 tools
# Source: https://github.com/makenotion/notion-mcp-server
# ---------------------------------------------------------------------------

_NOTION_TOOLS: list[dict] = [
    {
        "name": "notion_search",
        "description": "Search across all pages and databases in the Notion workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text."},
                "filter": {
                    "type": "object",
                    "description": "Filter results by object type (page or database).",
                },
                "sort": {"type": "object", "description": "Sort order for results."},
                "page_size": {"type": "number", "description": "Number of results to return."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_a_page",
        "description": "Retrieve the properties of a specific Notion page by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "The Notion page ID."},
            },
            "required": ["page_id"],
        },
    },
    {
        "name": "create_a_page",
        "description": "Create a new page in Notion under a specified parent page or database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent": {"type": "object", "description": "Parent page or database reference."},
                "properties": {"type": "object", "description": "Page properties."},
                "children": {"type": "array", "description": "Page content blocks."},
            },
            "required": ["parent", "properties"],
        },
    },
    {
        "name": "update_page_properties",
        "description": "Update properties of an existing Notion page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
                "properties": {"type": "object", "description": "Properties to update."},
            },
            "required": ["page_id", "properties"],
        },
    },
    {
        "name": "archive_page",
        "description": "Archive (soft-delete) a Notion page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
            },
            "required": ["page_id"],
        },
    },
    {
        "name": "query_database",
        "description": "Query a Notion database with optional filters, sorts, and pagination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "database_id": {"type": "string", "description": "The database ID."},
                "filter": {"type": "object", "description": "Filter conditions."},
                "sorts": {"type": "array", "description": "Sort criteria."},
                "page_size": {"type": "number"},
                "start_cursor": {"type": "string"},
            },
            "required": ["database_id"],
        },
    },
    {
        "name": "retrieve_a_database",
        "description": "Retrieve metadata and schema of a Notion database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "database_id": {"type": "string"},
            },
            "required": ["database_id"],
        },
    },
    {
        "name": "create_a_database",
        "description": "Create a new database in Notion under a specified parent page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent": {"type": "object"},
                "title": {"type": "array", "description": "Database title."},
                "properties": {"type": "object", "description": "Database property schema."},
            },
            "required": ["parent", "title", "properties"],
        },
    },
    {
        "name": "update_a_database",
        "description": "Update the title or properties schema of an existing Notion database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "database_id": {"type": "string"},
                "title": {"type": "array"},
                "properties": {"type": "object"},
            },
            "required": ["database_id"],
        },
    },
    {
        "name": "retrieve_block_children",
        "description": "Retrieve the child blocks of a specific Notion block or page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "block_id": {"type": "string", "description": "The block or page ID."},
                "page_size": {"type": "number"},
                "start_cursor": {"type": "string"},
            },
            "required": ["block_id"],
        },
    },
    {
        "name": "append_block_children",
        "description": "Append new content blocks as children of a Notion block or page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "block_id": {"type": "string"},
                "children": {"type": "array", "description": "Array of block objects to append."},
            },
            "required": ["block_id", "children"],
        },
    },
    {
        "name": "delete_a_block",
        "description": "Delete (archive) a specific Notion block.",
        "input_schema": {
            "type": "object",
            "properties": {
                "block_id": {"type": "string"},
            },
            "required": ["block_id"],
        },
    },
    {
        "name": "create_comment",
        "description": "Create a new comment on a Notion page or in a discussion thread.",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent": {"type": "object", "description": "Page reference for the comment."},
                "discussion_id": {"type": "string"},
                "rich_text": {"type": "array", "description": "Comment content as rich text."},
            },
            "required": ["rich_text"],
        },
    },
    {
        "name": "retrieve_comments",
        "description": "Retrieve comments from a Notion page or block.",
        "input_schema": {
            "type": "object",
            "properties": {
                "block_id": {"type": "string", "description": "Page or block ID."},
                "page_size": {"type": "number"},
                "start_cursor": {"type": "string"},
            },
            "required": ["block_id"],
        },
    },
    {
        "name": "move_page",
        "description": "Move a Notion page to a different parent page or database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string"},
                "parent": {"type": "object", "description": "New parent reference."},
            },
            "required": ["page_id", "parent"],
        },
    },
]


# ---------------------------------------------------------------------------
# Combined corpus
# ---------------------------------------------------------------------------

BASE_TOOLS: list[dict] = (
    _CHROME_DEVTOOLS_TOOLS
    + _GITHUB_TOOLS
    + _PLAYWRIGHT_TOOLS
    + _FILESYSTEM_TOOLS
    + _GIT_TOOLS
    + _NOTION_TOOLS
)
"""All ~138 real MCP server tool definitions."""


def generate_tools(n: int) -> list[dict]:
    """Return *n* tool definitions.

    * If *n* ≤ len(BASE_TOOLS), slices the corpus.
    * If *n* > len(BASE_TOOLS), duplicates tools with environment prefixes
      (``staging_``, ``internal_``, ``dev_``, …) until we reach *n*.
    """
    if n <= len(BASE_TOOLS):
        return BASE_TOOLS[:n]

    prefixes = ["staging_", "internal_", "dev_", "test_", "preview_"]
    tools = list(BASE_TOOLS)
    pi = 0
    while len(tools) < n:
        prefix = prefixes[pi % len(prefixes)]
        for tool in BASE_TOOLS:
            if len(tools) >= n:
                break
            dup = dict(tool)
            dup["name"] = prefix + tool["name"]
            dup["description"] = f"[{prefix.rstrip('_')}] {tool['description']}"
            tools.append(dup)
        pi += 1
    return tools[:n]


# ---------------------------------------------------------------------------
# Ground truth: (query, expected_tool_names)
# ---------------------------------------------------------------------------

GROUND_TRUTH: list[tuple[str, list[str]]] = [
    # -- Screenshot / visual --
    (
        "take a screenshot of the page",
        ["take_screenshot", "browser_take_screenshot"],
    ),
    (
        "capture an accessibility snapshot",
        ["take_snapshot", "browser_snapshot"],
    ),
    # -- Navigation --
    (
        "navigate to a URL",
        ["navigate_page", "browser_navigate", "new_page"],
    ),
    (
        "go back to the previous page",
        ["go_back", "browser_navigate_back"],
    ),
    # -- Click / Interaction --
    (
        "click on an element",
        ["click", "browser_click"],
    ),
    (
        "type text into an input field",
        ["type_text", "browser_type"],
    ),
    (
        "press the Enter key",
        ["press_key", "browser_press_key"],
    ),
    (
        "select an option from a dropdown",
        ["select_option", "browser_select_option"],
    ),
    (
        "hover over a button",
        ["hover", "browser_hover"],
    ),
    # -- File operations --
    (
        "list files in a directory",
        ["list_directory", "list_directory_with_sizes", "search_files"],
    ),
    (
        "read the contents of a file",
        ["read_file", "read_text_file", "read_multiple_files", "get_file_contents"],
    ),
    (
        "create a new file with content",
        ["write_file", "create_or_update_file"],
    ),
    (
        "move or rename a file",
        ["move_file", "move_page"],
    ),
    (
        "search for files matching a pattern",
        ["search_files", "search_code"],
    ),
    # -- Git operations --
    (
        "show git status of the repo",
        ["git_status"],
    ),
    (
        "view the diff of unstaged changes",
        ["git_diff_unstaged", "git_diff"],
    ),
    (
        "commit staged changes with a message",
        ["git_commit"],
    ),
    (
        "create a new branch",
        ["git_create_branch", "create_branch"],
    ),
    (
        "view the git commit log history",
        ["git_log", "list_commits"],
    ),
    # -- GitHub-specific --
    (
        "create a pull request",
        ["create_pull_request"],
    ),
    (
        "list open issues in the repository",
        ["list_issues"],
    ),
    (
        "search for code across repositories",
        ["search_code"],
    ),
    (
        "run a GitHub Actions workflow",
        ["run_workflow"],
    ),
    (
        "merge a pull request",
        ["merge_pull_request"],
    ),
    # -- Notion --
    (
        "query a database with filters",
        ["query_database"],
    ),
    (
        "search for pages in the workspace",
        ["notion_search"],
    ),
    (
        "create a comment on a page",
        ["create_comment"],
    ),
    (
        "append content blocks to a page",
        ["append_block_children"],
    ),
    # -- Cross-server --
    (
        "evaluate JavaScript on the page",
        ["evaluate_script", "browser_evaluate"],
    ),
    (
        "list network requests from the page",
        ["list_network_requests", "browser_network_requests"],
    ),
]
