import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import CatBoostPage from './pages/CatBoostPage';
import ProphetPage from './pages/ProphetPage';

export default function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/catboost" element={<CatBoostPage />} />
            <Route path="/prophet" element={<ProphetPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}
