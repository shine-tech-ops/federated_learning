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
          <motion.span 
            className={styles.badge}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.3 }}
          >
            🌸 Open Source Federated Learning
          </motion.span>
          
          <h1 className={styles.title}>
            Build <span className={styles.highlight}>Federated AI</span> with Flower
          </h1>
          
          <p className={styles.subtitle}>
            A friendly federated learning framework. Train AI models collaboratively 
            across decentralized data while keeping data private and secure.
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
              href="https://github.com/adap/flower" 
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
              <span className={styles.codeTitle}>quickstart.py</span>
            </div>
            <pre className={styles.code}>
              <code>{`import flwr as fl

# Define your Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train)
        return model.get_weights(), len(x_train), {}

# Start Flower client
fl.client.start_client(
    server_address="localhost:8080",
    client=FlowerClient()
)`}</code>
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
