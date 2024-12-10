import React from 'react';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import BuildingInfo from './components/BuildingInfo';
import CommentsSection from './components/CommentsSection';
import Footer from './components/Footer';
import './App.css';

function App() {
  return (
    <div className="App">
      <Header />
      <main>
        <UploadSection />
        <BuildingInfo />
        <CommentsSection />
      </main>
      <Footer />
    </div>
  );
}

export default App;
