# Quickstart Guide: AI/Spec-Driven Technical Book with Docusaurus

## Prerequisites

- Node.js 20+ installed
- npm or yarn package manager
- Git for version control
- GitHub account for deployment

## Setup Instructions

### 1. Clone and Initialize the Repository

```bash
# Clone your repository
git clone <your-repo-url>
cd <your-repo-name>

# Install dependencies
npm install
# OR
yarn install
```

### 2. Install Docusaurus

```bash
# Create a new Docusaurus site (if not already done)
npm init docusaurus@latest website classic
# OR if updating existing
npm install @docusaurus/core@latest @docusaurus/module-type-aliases@latest
```

### 3. Project Structure Setup

The project follows this structure for optimal modularity:

```
website/
├── docs/                 # All book content
│   ├── intro.md
│   ├── quarter-overview.md
│   ├── module-1-ros2/
│   ├── module-2-digital-twin/
│   ├── module-3-isaac/
│   ├── module-4-vla/
│   ├── capstone/
│   ├── appendix.md
│   └── glossary.md
├── static/
│   └── img/             # All diagrams and images
├── src/
│   └── components/      # Custom components
├── docusaurus.config.js # Site configuration
└── sidebars.js          # Navigation configuration
```

### 4. Start Development Server

```bash
# Start local development server
npm run start
# OR
yarn start

# The site will be available at http://localhost:3000
```

### 5. Build and Deploy

```bash
# Build static files for production
npm run build
# OR
yarn build

# Deploy to GitHub Pages
npm run deploy
# OR
yarn deploy
```

## Creating New Content

### Adding a New Chapter

1. Create a new markdown file in the appropriate module directory:

```markdown
<!-- docs/module-1-ros2/new-chapter.md -->
---
title: Chapter Title
sidebar_label: Chapter Title
description: Brief description of the chapter
---

# Chapter Title

Content goes here...

## Section

More content...
```

2. Update `sidebars.js` to include the new chapter in navigation:

```javascript
// sidebars.js
module.exports = {
  tutorial: [
    {
      type: 'category',
      label: 'Module 1: ROS 2',
      items: [
        'module-1-ros2/index',
        'module-1-ros2/new-chapter',  // Add this line
      ],
    },
  ],
};
```

### Adding Code Examples

Use Docusaurus' built-in code block syntax with syntax highlighting:

```markdown
## Code Example

Here's a Python example:

```python
def example_function():
    """This is an example function."""
    return "Hello, Physical AI!"
```

Or a C++ example:

```cpp
#include <ros2/ros2.hpp>

int main() {
    // Your ROS 2 code here
    return 0;
}
```
```

### Adding Diagrams and Images

1. Place images in the appropriate module folder under `static/img/`
2. Reference them in markdown:

```markdown
![Description of diagram](/img/module-1/diagram-name.png)
```

## Configuration Settings

### Docusaurus Configuration (`docusaurus.config.js`)

Key settings for the technical book:

```javascript
module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Bridging AI cognition with physical embodiment',
  url: 'https://your-username.github.io',
  baseUrl: '/ai-book/',
  organizationName: 'your-username',
  projectName: 'ai-book',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  // ... other configuration
};
```

### Navigation Configuration (`sidebars.js`)

Organize content by modules:

```javascript
module.exports = {
  sidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro', 'quarter-overview'],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 – The Robotic Nervous System',
      items: [
        'module-1-ros2/index',
        'module-1-ros2/concepts',
        // ... more items
      ],
    },
    // ... more modules
  ],
};
```

## Deployment to GitHub Pages

1. Ensure your repository is set up for GitHub Pages
2. Configure deployment settings in GitHub repository:
   - Settings → Pages → Source: Deploy from a branch
   - Branch: gh-pages, / (root)

3. Run deployment command:
```bash
GIT_USER=<Your GitHub username> \
  CURRENT_BRANCH=main \
  USE_SSH=true \
  npm run deploy
```

## Quality Assurance Checks

Before publishing content:

- [ ] All links are valid and not broken
- [ ] Code examples are properly formatted and functional
- [ ] Images load correctly
- [ ] Navigation works properly between modules
- [ ] Search functionality works
- [ ] Content follows educational clarity principles
- [ ] Terminology is consistent across modules