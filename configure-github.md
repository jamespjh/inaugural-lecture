# Configure GitHub Pages for Slideshow Deployment

After the `Build and deploy slides` workflow is merged to main and runs successfully, follow these steps to enable GitHub Pages in the repository settings.

## Steps

1. **Navigate to Repository Settings**
   - Go to your GitHub repository
   - Click the **Settings** tab (top navigation bar)
   - Scroll to and click **Pages** in the left sidebar

2. **Configure Pages Source**
   - Under "Build and deployment" section, look for "Source"
   - Select **Deploy from a branch** dropdown (if visible)
   - Change it to **GitHub Actions**
   - The dropdown will disappear once GitHub Actions is selected

3. **Verify Deployment**
   - You should see a green checkmark message: "Your site is live at `https://<username>.github.io/<repo>`"
   - It may take a few moments for the URL to appear
   - If you see "Deployment pending", wait a few seconds and refresh

4. **Test the Deployed Site**
   - Click the provided URL or manually visit it
   - Verify that:
     - All slides load and render correctly
     - CSS styling (colors, logo) displays as expected
     - Videos and images in slides are visible
     - Navigation (arrow keys, menu) works

5. **Optional: Custom Domain (Later)**
   - If you want to configure your UCL cloud domain later, you can add it here
   - For now, leave it empty to use the default `github.io` domain

## Troubleshooting

- **Pages settings show "Deploy from a branch" only**: The workflow may not have created an artifact yet. Wait for the workflow to run and complete, then refresh the settings page.
- **Site not deploying**: Check the **Actions** tab to see if the workflow ran successfully. Look for errors in the "Build slides" or "Deploy to GitHub Pages" steps.
- **Assets (images/videos) not loading**: This may indicate an asset path issue in the Quarto configuration. The workflow logs will show any build warnings.
