import { motion } from 'framer-motion'
import styles from './Features.module.css'

const features = [
  {
    icon: (
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <path d="M9 12l2 2 4-4"/>
      </svg>
    ),
    title: 'Privacy-Preserving',
    description: 'Keep data on-device. Train models collaboratively without sharing raw data, ensuring compliance with privacy regulations.',
    color: '#10B981',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="5" r="2"/>
        <circle cx="5" cy="12" r="2"/>
        <circle cx="19" cy="12" r="2"/>
        <circle cx="12" cy="19" r="2"/>
        <line x1="12" y1="7" x2="12" y2="17"/>
        <line x1="7" y1="12" x2="17" y2="12"/>
        <line x1="8.5" y1="8.5" x2="15.5" y2="15.5"/>
        <line x1="15.5" y1="8.5" x2="8.5" y2="15.5"/>
      </svg>
    ),
    title: 'Hierarchical Network',
    description: 'Multi-tier architecture with Central Server, Regional Nodes, and Edge Devices. Supports RabbitMQ, MQTT, and gRPC protocols for efficient communication.',
    color: '#3B82F6',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
      </svg>
    ),
    title: 'Highly Scalable',
    description: 'Horizontally scalable architecture. Easily add regional nodes and edge devices. Scale from a few devices to thousands with minimal overhead.',
    color: '#F59E0B',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
        <line x1="8" y1="21" x2="16" y2="21"/>
        <line x1="12" y1="17" x2="12" y2="21"/>
        <path d="M7 8h10M7 12h10M7 16h6"/>
      </svg>
    ),
    title: 'Heterogeneous Models',
    description: 'Support for PyTorch, TensorFlow, ONNX, and more. Works with models of any size, from lightweight mobile models to large-scale transformers.',
    color: '#8B5CF6',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
      </svg>
    ),
    title: 'Auto Optimization',
    description: 'Multiple aggregation algorithms including FedAvg, FedProx, FedAdam, and more. Automatically optimize training strategies for better convergence and performance.',
    color: '#EC4899',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
        <polyline points="10 9 9 9 8 9"/>
        <path d="M12 11v6M9 14l3-3 3 3"/>
      </svg>
    ),
    title: 'Comprehensive Documentation',
    description: 'Complete documentation covering architecture, deployment, development guides, and API references. Get started quickly with step-by-step tutorials and examples.',
    color: '#06B6D4',
  },
]

export default function Features() {
  return (
    <section id="features" className={styles.features}>
      <div className={styles.container}>
        <motion.div 
          className={styles.header}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <span className={styles.label}>Features</span>
          <h2 className={styles.title}>Everything you need for federated learning</h2>
          <p className={styles.subtitle}>
            A comprehensive federated learning platform with hierarchical network architecture, 
            automatic optimization, and support for heterogeneous models across diverse devices.
          </p>
        </motion.div>
        
        <div className={styles.grid}>
          {features.map((feature, index) => (
            <motion.div 
              key={feature.title}
              className={styles.card}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div 
                className={styles.iconWrapper}
                style={{ backgroundColor: `${feature.color}15`, color: feature.color }}
              >
                {feature.icon}
              </div>
              <h3 className={styles.cardTitle}>{feature.title}</h3>
              <p className={styles.cardDescription}>{feature.description}</p>
            </motion.div>
          ))}
        </div>
        
        {/* Call to action */}
        <motion.div 
          className={styles.ctaSection}
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <div className={styles.ctaCard}>
            <div className={styles.ctaContent}>
              <h3 className={styles.ctaTitle}>Ready to get started?</h3>
              <p className={styles.ctaText}>
                Start building federated learning applications with our comprehensive platform today.
              </p>
            </div>
            <div className={styles.ctaActions}>
              <a href="#quickstart" className={styles.ctaPrimary}>
                Get Started
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
              </a>
              <a 
                href="https://flower.ai/docs" 
                target="_blank" 
                rel="noopener noreferrer"
                className={styles.ctaSecondary}
              >
                View Documentation
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
