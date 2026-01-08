import { motion } from 'framer-motion'
import styles from './Hero.module.css'

export default function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.container}>
        <motion.div 
          className={styles.content}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
        
          
          <h1 className={styles.title}>
            Learn Local <span className={styles.highlight}>Evolve Global</span> 
          </h1>
          
          <p className={styles.subtitle}>
            Empower distributed intelligence. Train models locallyon edge devices, aggregate insights globally , federated cnn training ,federated optimization, federated large-small model
          </p>
          
          <div className={styles.actions}>
            <motion.a 
              href="#quickstart" 
              className={styles.primaryBtn}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              Get Started
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </motion.a>
            
            <motion.a 
              href="https://gitlab.osvie.com/shinetechzz/federated_learning" 
              target="_blank"
              rel="noopener noreferrer"
              className={styles.secondaryBtn}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              ⭐ Star on GitHub
            </motion.a>
          </div>
          
        </motion.div>
        
        <motion.div 
          className={styles.visual}
          initial={{ opacity: 0, x: 40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
        >
          <div className={styles.codeBlock}>
            <div className={styles.codeHeader}>
              <span className={styles.dot} style={{ background: '#FF5F56' }} />
              <span className={styles.dot} style={{ background: '#FFBD2E' }} />
              <span className={styles.dot} style={{ background: '#27C93F' }} />
              <span className={styles.codeTitle}>device_example.py</span>
            </div>
            <pre className={styles.code}>
              <code>{`from device import EdgeDevice

# Initialize edge device
device = EdgeDevice(
    device_id="device_001",
    mqtt_config={
        "host": "localhost",
        "port": 1883
    }
)

# Start federated learning
device.start()

# Device automatically:
# - Connects to region node
# - Receives training tasks
# - Trains model locally
# - Reports metrics globally`}</code>
            </pre>
          </div>
          
          {/* Floating elements */}
          <motion.div 
            className={styles.floatingCard}
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          >
            <span className={styles.floatingIcon}>🔐</span>
            <span>Privacy-First</span>
          </motion.div>
          
          <motion.div 
            className={`${styles.floatingCard} ${styles.floatingCard2}`}
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
          >
            <span className={styles.floatingIcon}>⚡</span>
            <span>Fast & Scalable</span>
          </motion.div>
        </motion.div>
      </div>
      
      {/* Background decoration */}
      <div className={styles.bgGradient} />
    </section>
  )
}
