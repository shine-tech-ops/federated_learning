import { motion } from 'framer-motion'
import styles from './QuickStart.module.css'

const steps = [
  {
    step: '01',
    title: 'Install Flower',
    description: 'Install Flower via pip with a single command.',
    code: 'pip install flwr',
  },
  {
    step: '02',
    title: 'Create a Client',
    description: 'Define your federated learning client with your model.',
    code: `import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # Train your model here
        return updated_params, num_samples, {}`,
  },
  {
    step: '03',
    title: 'Start Training',
    description: 'Launch the server and connect your clients.',
    code: `# Start server
fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))

# Start client
fl.client.start_client(server_address="localhost:8080", client=FlowerClient())`,
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
            Get started with federated learning in just three simple steps. 
            No complex setup required.
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
