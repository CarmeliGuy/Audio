# --- Update existing GitHub project (clean version) ---
param(
    [string]$commitMessage = "Update project"
)

Write-Host "Adding modified files..."
git add .

Write-Host "Committing changes..."
git commit -m $commitMessage

Write-Host "Pulling latest changes..."
git pull --rebase origin main

Write-Host "Pushing to GitHub..."
git push origin main

Write-Host "Project updated successfully!"
