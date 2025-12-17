import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Hero from './components/Hero'
import QuickStart from './components/QuickStart'
import Features from './components/Features'
import Footer from './components/Footer'
import Documentation from './components/Documentation'

function HomePage() {
  return (
    <>
      <main>
        <Hero />
        <QuickStart />
        <Features />
      </main>
      <Footer />
    </>
  )
}

export default function App() {
  return (
    <Router>
      <Nav />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/docs" element={<Documentation />} />
        <Route path="/docs/:docId" element={<Documentation />} />
      </Routes>
    </Router>
  )
}
