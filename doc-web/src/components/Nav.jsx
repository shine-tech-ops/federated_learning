import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import styles from './Nav.module.css'

const navLinks = [
  { label: 'Home', href: '/' },
  { label: 'Docs', href: '/docs' },
  { label: 'Examples', href: '#examples' },
]

export default function Nav() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  return (
    <motion.nav 
      className={styles.nav}
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className={styles.container}>
        {/* Logo */}
        <Link to="/" className={styles.logo}>
          <svg viewBox="0 0 64 64" width="40" height="40">
            <circle cx="32" cy="32" r="28" fill="#FFC107"/>
            <path d="M32 12 L36 28 L32 32 L28 28 Z" fill="#111827"/>
            <path d="M32 52 L28 36 L32 32 L36 36 Z" fill="#111827"/>
            <path d="M12 32 L28 28 L32 32 L28 36 Z" fill="#111827"/>
            <path d="M52 32 L36 36 L32 32 L36 28 Z" fill="#111827"/>
            <circle cx="32" cy="32" r="6" fill="#8B5CF6"/>
          </svg>
          <span className={styles.logoText}>FedAvg</span>
        </Link>

        {/* Desktop Links */}
        <div className={styles.desktopLinks}>
          {navLinks.map((link) => (
            link.href.startsWith('#') ? (
              <a key={link.label} href={link.href} className={styles.navLink}>
                {link.label}
              </a>
            ) : (
              <Link 
                key={link.label} 
                to={link.href} 
                className={`${styles.navLink} ${location.pathname === link.href ? styles.active : ''}`}
              >
                {link.label}
              </Link>
            )
          ))}
        </div>

        {/* Desktop Actions */}
        <div className={styles.actions}>
          <a 
            href="https://github.com/adap/flower" 
            target="_blank" 
            rel="noopener noreferrer"
            className={styles.githubBtn}
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            <span>Star</span>
          </a>
        </div>

        {/* Mobile Menu Button */}
        <button 
          className={styles.mobileMenuBtn}
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
            {mobileOpen ? (
              <path d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div 
            className={styles.mobileMenu}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
          >
            {navLinks.map((link) => (
              link.href.startsWith('#') ? (
                <a 
                  key={link.label} 
                  href={link.href} 
                  className={styles.mobileNavLink}
                  onClick={() => setMobileOpen(false)}
                >
                  {link.label}
                </a>
              ) : (
                <Link 
                  key={link.label} 
                  to={link.href} 
                  className={`${styles.mobileNavLink} ${location.pathname === link.href ? styles.active : ''}`}
                  onClick={() => setMobileOpen(false)}
                >
                  {link.label}
                </Link>
              )
            ))}
            <a 
              href="https://github.com/adap/flower" 
              target="_blank" 
              rel="noopener noreferrer"
              className={styles.mobileGithubBtn}
            >
              ⭐ Star on GitHub
            </a>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}
