// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
    },
    {
      type: 'doc',
      id: 'quarter-overview',
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 – The Robotic Nervous System',
      items: [
        'module-1-ros2/index',
        'module-1-ros2/concepts',
        'module-1-ros2/setup',
        'module-1-ros2/exercises',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twins (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/index',
        'module-2-digital-twin/concepts',
        'module-2-digital-twin/tools',
        'module-2-digital-twin/exercises',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: AI Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-isaac/index',
        'module-3-isaac/concepts',
        'module-3-isaac/implementation',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems',
      items: [
        'module-4-vla/index',
        'module-4-vla/concepts',
        'module-4-vla/applications',
        'module-4-vla/exercises',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Capstone Project: Autonomous Humanoid Robot',
      items: [
        'capstone/index',
        'capstone/requirements',
        'capstone/implementation',
        'capstone/evaluation',
      ],
      collapsed: false,
    },
    {
      type: 'doc',
      id: 'summary',
    },
    {
      type: 'doc',
      id: 'glossary',
    },
    {
      type: 'doc',
      id: 'appendix',
    },
  ],
};

module.exports = sidebars;