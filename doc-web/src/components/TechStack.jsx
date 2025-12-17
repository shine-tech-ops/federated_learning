import { motion } from 'framer-motion'
import styles from './TechStack.module.css'

const techStack = [
  // Frontend
  { name: 'React', icon: '⚛️', color: '#61DAFB', category: 'Frontend' },
  { name: 'Vue 3', icon: '💚', color: '#42B883', category: 'Frontend' },
  { name: 'Vite', icon: '⚡', color: '#646CFF', category: 'Frontend' },
  { name: 'TypeScript', icon: '📘', color: '#3178C6', category: 'Frontend' },
  { name: 'Element Plus', icon: '🎨', color: '#409EFF', category: 'Frontend' },
  { name: 'TailwindCSS', icon: '🎨', color: '#06B6D4', category: 'Frontend' },
  
  // Backend
  { name: 'Django', icon: '🎯', color: '#092E20', category: 'Backend' },
  { name: 'Python', icon: '🐍', color: '#3776AB', category: 'Backend' },
  { name: 'PostgreSQL', icon: '🐘', color: '#4169E1', category: 'Backend' },
  { name: 'Redis', icon: '🔴', color: '#DC382D', category: 'Backend' },
  
  // ML & AI
  { name: 'PyTorch', icon: '🔥', color: '#EE4C2C', category: 'ML' },
  { name: 'NumPy', icon: '🔢', color: '#013243', category: 'ML' },
  
  // Message Queue
  { name: 'RabbitMQ', icon: '🐰', color: '#FF6600', category: 'Queue' },
  { name: 'MQTT', icon: '📡', color: '#660066', category: 'Queue' },
  
  // Storage
  { name: 'MinIO', icon: '💾', color: '#C72E49', category: 'Storage' },
  
  // DevOps
  { name: 'Docker', icon: '🐳', color: '#2496ED', category: 'DevOps' },
]

export default function TechStack() {
  // 复制数组以实现无限滚动效果
  const duplicatedStack = [...techStack, ...techStack]

  return (
    <section className={styles.techStack}>
      <div className={styles.container}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className={styles.header}
        >
          <h2 className={styles.title}>Built with Modern Tech Stack</h2>
          <p className={styles.subtitle}>
            Powered by cutting-edge technologies for distributed machine learning
          </p>
        </motion.div>

        <div className={styles.scrollContainer}>
          <motion.div
            className={styles.scrollTrack}
            animate={{
              x: ['0%', '-50%'],
            }}
            transition={{
              x: {
                repeat: Infinity,
                repeatType: 'loop',
                duration: 30,
                ease: 'linear',
              },
            }}
          >
            {duplicatedStack.map((tech, index) => (
              <div
                key={`${tech.name}-${index}`}
                className={styles.techCard}
                style={{ '--tech-color': tech.color }}
              >
                <span className={styles.techIcon}>{tech.icon}</span>
                <span className={styles.techName}>{tech.name}</span>
                <span className={styles.techCategory}>{tech.category}</span>
              </div>
            ))}
          </motion.div>
        </div>

        {/* Gradient overlays for fade effect */}
        <div className={styles.gradientLeft}></div>
        <div className={styles.gradientRight}></div>
      </div>
    </section>
  )
}
