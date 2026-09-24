---
name: ship
description: Ships current uncommitted changes by committing them on the WIP branch, pushing to origin, opening or reusing a pull request into main, and merging it. Use only when the user explicitly invokes /ship. Never invoke automatically.
disable-model-invocation: true
---

# Ship

When I invoke `/ship`, follow this workflow:

1. Inspect the current Git state:
   - Run `git status`
   - Inspect the current diff
   - Understand what changes have been made

2. Make sure there are changes to ship.
   - If there are no changes, stop and report that there are no changes to ship.

3. Never discard, reset, overwrite, clean, or otherwise remove any of my uncommitted changes.

4. Ensure that the current branch is `WIP`.
   - If I am already on `WIP`, continue.
   - If I am on another branch and `WIP` does not exist, create `WIP` and continue.
   - If I am on another branch and `WIP` already exists, switch to `WIP`.
   - Do NOT create feature branches.
   - Do NOT create any other branches.
   - Never switch branches in a way that could discard or overwrite my uncommitted changes. If switching is unsafe, stop and report the issue.

5. Do NOT run tests, linting, type checking, or any other project validation.
   - This workflow is only for shipping the existing changes.

6. Stage the intended changes.

7. Create a concise and meaningful commit message based on the actual changes.

8. Commit the changes.

9. Push the `WIP` branch to `origin`.
   - Never force push.

10. Create a GitHub Pull Request from:
    `WIP` → `main`

    - Generate a concise PR title based on the changes.
    - Generate a short useful PR description.
    - Do not create a draft PR.
    - Before creating a PR, check whether a PR from `WIP` to `main` already exists.
    - If one already exists, do not create a duplicate. Use the existing PR.

11. Merge the Pull Request into `main`.
    - Do not force merge.
    - Do not bypass branch protection.
    - Do not bypass required approvals or checks.
    - If GitHub prevents the merge, stop and report why.

12. Use GitHub CLI (`gh`) for GitHub operations where appropriate.

13. At the end, always return the actual result of the workflow.

If successful, report:
- Branch: WIP
- Commit hash
- PR number
- PR URL
- Confirmation that WIP was merged into main

If unsuccessful, report:
- Which step failed
- The reason for failure
- PR URL if a PR was created
- Confirm that no changes were discarded

Important restrictions:
- No feature branches
- No tests or validation
- No force push
- No force merge
- No destructive Git commands
- No discarding user changes
- No duplicate PRs
- No automatic invocation

The purpose of `/ship` is simply:

current changes → WIP → commit → push → PR → merge → report result
