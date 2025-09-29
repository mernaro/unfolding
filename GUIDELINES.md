# Contributing to Unfolding: Super Resolution and Image Segmentation

Thank you for contributing! To keep the project organized and maintainable, please follow these guidelines.

---

## Commit Message Format

We follow the **Conventional Commits** standard:

`<type>(<scope>): <description>`

### Types

- **feat**: A new feature  
- **fix**: A bug fix  
- **docs**: Documentation changes  
- **chore**: Maintenance tasks (no code change)  
- **test**: Adding or updating tests  
- **refactor**: Code changes that neither fixes a bug nor adds a feature  
- **perf**: Performance improvements  
- **config**: Changes to configuration files

### Scope

Optional, indicates the module or folder affected, e.g., `src`, `notebooks`, `experiments`.

### Description

Short summary of your changes (max ~50 characters).

### Examples

```text
feat(src): add training script
docs: update README.md
config(experiments): update default hyperparameters

```


---

## Branching

- Use **feature branches** for new features:  
`git checkout -b feature/add-dataloader`

- Use **bugfix branches** for fixes:  
`git checkout -b fix/train-script-bug`

- Merge into `main` via Pull Request only. Ensure your branch is up-to-date with `main` before merging:  
`git pull origin main`


---

Thanks for helping keep this project clean and organized! 
