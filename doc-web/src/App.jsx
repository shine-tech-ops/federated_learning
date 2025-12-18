import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Hero from './components/Hero'
import QuickStart from './components/QuickStart'
import Features from './components/Features'
import TechStack from './components/TechStack'
import Footer from './components/Footer'
import Documentation from './components/Documentation'
import Examples from './components/Examples'

function HomePage() {
  return (
    <>
      <main>
        <Hero />
        <TechStack />

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
        <Route path="/examples" element={<Examples />} />
        <Route path="/examples/:exampleId" element={<Examples />} />
      </Routes>
    </Router>
  )
}
