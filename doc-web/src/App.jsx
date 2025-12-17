import Nav from './components/Nav'
import Hero from './components/Hero'
import QuickStart from './components/QuickStart'
import Features from './components/Features'
import Footer from './components/Footer'

export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <QuickStart />
        <Features />
      </main>
      <Footer />
    </>
  )
}
