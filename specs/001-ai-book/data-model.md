# Data Model: AI/Spec-Driven Technical Book with Docusaurus

## Content Entities

### Book Module
- **Description**: Represents a major section of the technical book (e.g., Module 1: ROS 2, Module 2: Digital Twins)
- **Fields**:
  - moduleId: string (unique identifier for the module)
  - title: string (display title of the module)
  - description: string (brief description of the module content)
  - order: number (sequence order in the curriculum)
  - prerequisites: string[] (module IDs that should be completed before this one)
  - learningObjectives: string[] (specific learning objectives for the module)
  - duration: string (estimated time to complete the module)
  - status: enum (draft, review, published)

### Chapter
- **Description**: Individual content sections within modules that cover specific topics
- **Fields**:
  - chapterId: string (unique identifier for the chapter)
  - moduleId: string (reference to the parent module)
  - title: string (display title of the chapter)
  - description: string (brief description of the chapter content)
  - order: number (sequence order within the module)
  - contentPath: string (file path to the markdown content)
  - type: enum (concept, lab, exercise, walkthrough, theory)
  - estimatedReadingTime: number (in minutes)
  - requiresCode: boolean (indicates if code examples are included)

### Learning Resource
- **Description**: Supplementary materials including labs, exercises, diagrams, and code examples
- **Fields**:
  - resourceId: string (unique identifier for the resource)
  - moduleId: string (reference to the associated module)
  - chapterId: string (reference to the associated chapter, optional)
  - title: string (display title of the resource)
  - type: enum (diagram, lab, exercise, code-example, video, external-link)
  - filePath: string (path to the resource file)
  - description: string (brief description of the resource)
  - difficulty: enum (beginner, intermediate, advanced)
  - duration: string (estimated time to complete if applicable)

### Navigation Structure
- **Description**: Hierarchical organization of content that guides users through the learning path
- **Fields**:
  - navId: string (unique identifier for the navigation item)
  - title: string (display title in navigation)
  - type: enum (category, link, doc)
  - docId: string (reference to document if type is 'doc')
  - items: NavigationStructure[] (child navigation items)
  - collapsed: boolean (whether the category is collapsed by default)

## Content Relationships

- Book Module **contains** multiple Chapters
- Book Module **contains** multiple Learning Resources
- Chapter **may have** multiple Learning Resources
- Navigation Structure **references** Book Modules and Chapters

## Content Validation Rules

### Module Validation
- Module titles must be unique across the book
- Module order values must be sequential without gaps
- Prerequisites must reference existing modules
- Learning objectives must be specific and measurable

### Chapter Validation
- Chapter titles must be unique within each module
- Chapter order values must be sequential within each module
- Content paths must exist and be valid markdown files
- Estimated reading time must be between 5-60 minutes

### Learning Resource Validation
- Resource file paths must exist in the static directory
- Difficulty ratings must match the target audience level
- Resources must be associated with a valid module

## Content State Transitions

### Module States
- draft → review → published
- published → review (when updates are needed)

### Chapter States
- draft → review → published
- published → review (when updates are needed)

### Learning Resource States
- draft → review → published
- published → review (when updates are needed)