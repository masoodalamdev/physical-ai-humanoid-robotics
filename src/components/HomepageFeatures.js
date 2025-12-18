import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI & Embodied Intelligence',
    Svg: require('../../static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Learn how intelligence emerges from the interaction between an agent and its environment.
        This embodied approach to AI recognizes that physical interaction shapes cognitive development.
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    Svg: require('../../static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Explore the fascinating world of humanoid robots that can interact with the real world.
        From ROS 2 to Vision-Language-Action systems, master the complete stack.
      </>
    ),
  },
  {
    title: 'Complete Learning Path',
    Svg: require('../../static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Four comprehensive modules building towards a capstone project.
        From robotic nervous systems to AI brain implementations.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}