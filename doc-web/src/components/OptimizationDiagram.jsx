export default function OptimizationDiagram() {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
      borderRadius: '12px',
      padding: '40px',
      margin: '30px 0'
    }}>
      <svg
        viewBox="0 0 800 720"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          width: '100%',
          maxWidth: '800px',
          height: 'auto',
          margin: '0 auto',
          display: 'block'
        }}
      >
        {/* Definitions */}
        <defs>
          <linearGradient id="optGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#7C3AED', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#5B21B6', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="optGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#06B6D4', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#0891B2', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="optGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#EC4899', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#DB2777', stopOpacity: 1 }} />
          </linearGradient>
          <linearGradient id="optGrad4" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style={{ stopColor: '#EF4444', stopOpacity: 1 }} />
            <stop offset="100%" style={{ stopColor: '#DC2626', stopOpacity: 1 }} />
          </linearGradient>
          <filter id="optShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.3"/>
          </filter>
          <marker id="optArrowCyan" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#06B6D4" />
          </marker>
          <marker id="optArrowPink" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#EC4899" />
          </marker>
          <marker id="optArrowRed" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#EF4444" />
          </marker>
        </defs>

        {/* Title */}
        <text x="400" y="35" fontFamily="Arial, sans-serif" fontSize="22" fontWeight="bold" fill="white" textAnchor="middle">
          Federated Surrogate Optimization Architecture
        </text>

        {/* Center Server (Optimization Management) */}
        <g filter="url(#optShadow)">
          <rect x="250" y="70" width="300" height="110" rx="12" fill="url(#optGrad1)" />
          <text x="400" y="105" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Center Server</text>
          <text x="400" y="130" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">(Optimization Management)</text>
          <text x="400" y="150" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.85">Search Space Definition</text>
          <text x="400" y="167" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.75">Task Distribution &amp; Result Collection</text>
        </g>

        {/* Arrow: Central to Regional */}
        <line x1="400" y1="180" x2="400" y2="250" stroke="#06B6D4" strokeWidth="3" markerEnd="url(#optArrowCyan)" />
        <text x="455" y="220" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">Task (Search Space)</text>

        {/* Regional Node (Surrogate Aggregation) */}
        <g filter="url(#optShadow)">
          <rect x="200" y="270" width="400" height="140" rx="12" fill="url(#optGrad2)" />
          <text x="400" y="310" fontFamily="Arial, sans-serif" fontSize="20" fontWeight="bold" fill="white" textAnchor="middle">Regional Node</text>
          <text x="400" y="335" fontFamily="Arial, sans-serif" fontSize="14" fill="white" textAnchor="middle" opacity="0.9">(Surrogate Aggregation)</text>
          <text x="400" y="360" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.85">FedAvg Strategy</text>
          <text x="400" y="380" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" opacity="0.85">Global Surrogate Model Building</text>
          <text x="400" y="397" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.75">Python + Flower + Optimization Solver</text>
        </g>

        {/* Aggregation Return Arrow */}
        <path d="M 605 340 Q 680 340, 680 125 Q 680 80, 550 80"
              stroke="#EF4444"
              strokeWidth="2.5"
              fill="none"
              markerEnd="url(#optArrowRed)"
              strokeDasharray="5,5" />
        <text x="700" y="210" fontFamily="Arial, sans-serif" fontSize="12" fill="white" fontWeight="600">Aggregation</text>
        <text x="700" y="227" fontFamily="Arial, sans-serif" fontSize="11" fill="white" fontWeight="500">+ Weights</text>

        {/* Arrow: Regional to Devices */}
        <line x1="300" y1="410" x2="160" y2="480" stroke="#EC4899" strokeWidth="3" markerEnd="url(#optArrowPink)" />
        <line x1="500" y1="410" x2="640" y2="480" stroke="#EC4899" strokeWidth="3" markerEnd="url(#optArrowPink)" />
        <text x="210" y="450" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">Aggregation</text>
        <text x="570" y="450" fontFamily="Arial, sans-serif" fontSize="13" fill="white" fontWeight="600">Aggregation</text>

        {/* Edge Device 1 (Sampler) */}
        <g filter="url(#optShadow)">
          <rect x="35" y="500" width="250" height="180" rx="12" fill="url(#optGrad3)" />
          <text x="160" y="535" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Device 1</text>
          <text x="160" y="560" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" opacity="0.9">(Sampler)</text>

          <rect x="55" y="575" width="210" height="90" rx="6" fill="rgba(255,255,255,0.15)" />
          <text x="160" y="595" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" fontWeight="600">OptimizationTrainer</text>
          <text x="160" y="615" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Sample search space</text>
          <text x="160" y="632" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Evaluate objective function</text>
          <text x="160" y="649" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Train local surrogate</text>
          <text x="160" y="666" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.75">SurrogateNet (MLP)</text>
        </g>

        {/* Edge Device 2 (Sampler) */}
        <g filter="url(#optShadow)">
          <rect x="515" y="500" width="250" height="180" rx="12" fill="url(#optGrad3)" />
          <text x="640" y="535" fontFamily="Arial, sans-serif" fontSize="18" fontWeight="bold" fill="white" textAnchor="middle">Device 2</text>
          <text x="640" y="560" fontFamily="Arial, sans-serif" fontSize="13" fill="white" textAnchor="middle" opacity="0.9">(Sampler)</text>

          <rect x="535" y="575" width="210" height="90" rx="6" fill="rgba(255,255,255,0.15)" />
          <text x="640" y="595" fontFamily="Arial, sans-serif" fontSize="12" fill="white" textAnchor="middle" fontWeight="600">OptimizationTrainer</text>
          <text x="640" y="615" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Sample search space</text>
          <text x="640" y="632" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Evaluate objective function</text>
          <text x="640" y="649" fontFamily="Arial, sans-serif" fontSize="11" fill="white" textAnchor="middle" opacity="0.9">• Train local surrogate</text>
          <text x="640" y="666" fontFamily="Arial, sans-serif" fontSize="10" fill="white" textAnchor="middle" opacity="0.75">SurrogateNet (MLP)</text>
        </g>

        {/* Legend */}
        <g transform="translate(50, 710)">
          <text x="0" y="0" fontFamily="Arial, sans-serif" fontSize="12" fill="white" fontWeight="600">Data Flow:</text>
          <line x1="80" y1="-5" x2="120" y2="-5" stroke="#06B6D4" strokeWidth="2" />
          <text x="130" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Task Distribution</text>
          <line x1="230" y1="-5" x2="270" y2="-5" stroke="#EC4899" strokeWidth="2" />
          <text x="280" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Model to Devices</text>
          <line x1="390" y1="-5" x2="430" y2="-5" stroke="#EF4444" strokeWidth="2" strokeDasharray="5,5" />
          <text x="440" y="0" fontFamily="Arial, sans-serif" fontSize="11" fill="white">Surrogate Aggregation</text>
        </g>
      </svg>
    </div>
  )
}
