import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Calculator from './calculator'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
     <div className="App">
            <Calculator />
        </div>
    </>
  );
}

export default App;