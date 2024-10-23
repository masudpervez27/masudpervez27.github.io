---
title: "How to Create a Website Using the Jekyll Minimalist Theme and Host It on GitHub Pages for Free"
date: 2024-10-24
tags:
    - Jekyll
header:
image: /assets/images/headers/2023-02-05-blog-portfolio-with-mm-header.jpg
teaser: /assets/images/headers/2023-02-05-blog-portfolio-with-mm-header.jpg
---

# How to Create a Website Using the Jekyll Minimalist Theme and Host It on GitHub Pages for Free

Creating a website has never been easier, especially with tools like Jekyll and GitHub Pages. In this post, I'll walk you through the steps to build a stunning website using the Jekyll Minimalist theme and host it for free on GitHub Pages. Recently, I built my own portfolio website using this method, and I was astonished by the final result—it's sleek, functional, and a great showcase of my work!

## Why Jekyll and GitHub Pages?

**Jekyll** is a simple, blog-aware static site generator. It takes your content (written in Markdown) and transforms it into a static website. **GitHub Pages** allows you to host your Jekyll site directly from your GitHub repository for free. This makes it an ideal choice for personal projects, portfolios, and blogs.

## Getting Started

### Step 1: Prerequisites

Before you begin, make sure you have:

1. A GitHub account.
2. Basic knowledge of Git and command line (terminal) usage.
3. Ruby installed on your computer (check by running `ruby -v` in your terminal).

### Step 2: Install Jekyll

To install Jekyll, open your terminal and run the following commands:

```bash
gem install --user-install bundler jekyll
```
### Step 3: Install Jekyll

Create a new Jekyll site with the Minimalist theme by running:

```bash
jekyll new my-website --force
cd my-website
```

### Step 4: Configure Your Site

Open the _config.yml file in your favorite text editor. This file contains configuration settings for your Jekyll site. Update the following settings:

```yaml
title: My Portfolio
description: A showcase of my projects and skills.
url: "https://yourusername.github.io/my-website"
```

### Step 5: Choose the Minimalist Theme
You can either clone the Minimalist theme directly or add it to your existing Jekyll site. If you haven't created your site yet, you can find the Minimalist theme here.

To use it, add the theme to your _config.yml:

```yaml
theme: minimal
```

### Step 6: Customize Your Site
Now it's time to customize your site! Update the content in the _posts directory and create new Markdown files to add your portfolio pieces. For example, create a new post:

```markdown
---
layout: post
title: "My First Project"
date: 2024-10-24
---
This is a brief description of my first project. You can find the repository [here](https://github.com/yourusername/my-first-project).
```

### Step 7: Preview Your Site Locally
To see your changes live, run the following command in your terminal:

```bash
bundle exec jekyll serve
```

Open your browser and navigate to http://localhost:4000 to view your site.

### Step 8: Set Up GitHub Pages

1. **Create a New Repository**: Go to GitHub and create a new repository named `my-website` (or your preferred name).

2. **Push Your Code**: Initialize a Git repository in your project folder and push it to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/my-website.git
   git push -u origin master

3. **Configure GitHub Pages**: Go to the repository settings on GitHub, scroll down to the "GitHub Pages" section, and select the branch you want to use (usually main or master). Save the changes.

### Step 9: Visit Your Live Site
After a few minutes, your site will be live at https://yourusername.github.io/my-website. Check it out!

**Final Thoughts**
I was truly amazed by how my portfolio website turned out using this method. The Jekyll Minimalist theme provided a clean and elegant design that perfectly showcased my projects. Plus, hosting on GitHub Pages made the process incredibly seamless and cost-effective!

Feel free to reach out if you have any questions or need help with your Jekyll site. Happy building! 🚀