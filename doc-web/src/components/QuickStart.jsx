import { motion } from 'framer-motion'
import styles from './QuickStart.module.css'

const steps = [
  {
    step: '01',
    title: 'Deploy Central Server',
    description: 'Start the central server and all infrastructure services with Docker.',
    code: `# Load Docker images
docker load -i images.tar

# Start all services
docker-compose up -d

# Access frontend at http://localhost:8086`,
  },
  {
    step: '02',
    title: 'Start Regional Node',
    description: 'Deploy regional node to connect central server and edge devices.',
    code: `cd regional
pip install -r requirements.txt

# Configure environment
export REGION_ID=region-001
export RABBITMQ_HOST=localhost
export MQTT_BROKER_HOST=localhost

# Start regional node
python run.py`,
  },
  {
    step: '03',
    title: 'Connect Edge Devices',
    description: 'Launch edge devices to participate in federated learning.',
    code: `cd device
pip install -r requirements.txt

# Start device with ID and region
python start_device.py device_001 3 http://localhost:8085

# Device will automatically connect and wait for tasks`,
  },
]

export default function QuickStart() {
  return (
    <section id="quickstart" className={styles.quickstart}>
      <div className={styles.container}>
        <motion.div 
          className={styles.header}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <span className={styles.label}>Quick Start</span>
          <h2 className={styles.title}>Up and running in minutes</h2>
          <p className={styles.subtitle}>
            Deploy a complete federated learning system with central server, 
            regional nodes, and edge devices in three steps.
          </p>
        </motion.div>
        
        <div className={styles.steps}>
          {steps.map((item, index) => (
            <motion.div 
              key={item.step}
              className={styles.stepCard}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.15 }}
            >
              <div className={styles.stepHeader}>
                <span className={styles.stepNumber}>{item.step}</span>
                <h3 className={styles.stepTitle}>{item.title}</h3>
              </div>
              <p className={styles.stepDescription}>{item.description}</p>
              <div className={styles.codeBlock}>
                <div className={styles.codeHeader}>
                  <button 
                    className={styles.copyBtn}
                    onClick={() => navigator.clipboard.writeText(item.code)}
                    title="Copy to clipboard"
                  >
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="9" y="9" width="13" height="13" rx="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                  </button>
                </div>
                <pre className={styles.code}>
                  <code>{item.code}</code>
                </pre>
              </div>
            </motion.div>
          ))}
        </div>
        
        <motion.div 
          className={styles.cta}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <a href="#docs" className={styles.ctaBtn}>
            Read the Documentation
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </a>
        </motion.div>
      </div>
    </section>
  )
}
