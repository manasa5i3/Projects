import React, { useState } from 'react';
import './Calculator.css'; // Create Calculator.css file

function Calculator() {
    const [display, setDisplay] = useState('');

    const handleButtonClick = (value) => {
        setDisplay((prevDisplay) => prevDisplay + value);
    };

    const handleClear = () => {
        setDisplay('');
    };

    const handleCalculate = () => {
        try {
            setDisplay(String(eval(display)));
        } catch (error) {
            setDisplay('Error');
        }
    };

    return (
        <div className="calculator">
            <input type="text" className="display" value={display} readOnly />
            <div className="buttons">
                <button onClick={() => handleButtonClick('7')}>7</button>
                <button onClick={() => handleButtonClick('8')}>8</button>
                <button onClick={() => handleButtonClick('9')}>9</button>
                <button onClick={() => handleButtonClick('/')}>/</button>
                <button onClick={() => handleButtonClick('4')}>4</button>
                <button onClick={() => handleButtonClick('5')}>5</button>
                <button onClick={() => handleButtonClick('6')}>6</button>
                <button onClick={() => handleButtonClick('*')}>*</button>
                <button onClick={() => handleButtonClick('1')}>1</button>
                <button onClick={() => handleButtonClick('2')}>2</button>
                <button onClick={() => handleButtonClick('3')}>3</button>
                <button onClick={() => handleButtonClick('-')}>-</button>
                <button onClick={() => handleButtonClick('0')}>0</button>
                <button onClick={handleClear}>C</button>
                <button onClick={handleCalculate}>=</button>
                <button onClick={() => handleButtonClick('+')}>+</button>
                
            </div>
        </div>
    );
}

export default Calculator;