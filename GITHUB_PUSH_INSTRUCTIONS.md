# How to Push to a Specific GitHub Branch

Since your project is not yet a Git repository, follow these steps to initialize it and push to GitHub.

## 1. Initialize Git
Open your terminal in the project folder (`Agri Predictor`) and run:

```bash
git init
```

## 2. Add Files
Stage all your files for the first commit:

```bash
git add .
```

## 3. Commit
Save your changes locally:

```bash
git commit -m "Initial commit"
```

## 4. Add Remote Repository
Link your local project to your GitHub repository. Replace `<YOUR_REPO_URL>` with your actual GitHub repository link (e.g., `https://github.com/username/repo.git`).

```bash
git remote add origin <YOUR_REPO_URL>
```

## 5. Push to a Specific Branch
To push to a specific branch (e.g., `feature-branch` or `main`), run the following. Replace `<BRANCH_NAME>` with the name of the branch you want to push to.

```bash
# Create and switch to the branch (if it doesn't exist locally yet)
git checkout -b <BRANCH_NAME>

# Push to GitHub
git push -u origin <BRANCH_NAME>
```

### Example: Pushing to `main`
```bash
git checkout -b main
git push -u origin main
```

### Example: Pushing to `dev`
```bash
git checkout -b dev
git push -u origin dev
```
