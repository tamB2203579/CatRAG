import React from 'react'
import Window from '../components/Window/Window'
import { useState } from 'react';

function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen(prev => !prev);
  };
  return (
    <div>
        <Window isOpen={isSidebarOpen} onToggle={toggleSidebar}/>
    </div>
  )
}

export default Home