import React, { useEffect } from 'react'
import Home from './pages/Home'
import './index.css'
import WebFont from 'webfontloader'

const App = () => {
  useEffect(() => {
        WebFont.load({
          google: {
            families: [
            'K2D:vietnamese',
            'Readex Pro:vietnamese'
          ]
          }
        });
      }, []);
  return (
    <div>
      <Home/>
    </div>
  );
};

export default App