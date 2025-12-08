import './App.css'
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from './Pages/Home';
import GetByParams from './Pages/GetByParams';
import GetByDateRange from './Pages/GetByDateRange';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/params" element={<GetByParams />} />
        <Route path="/daterange" element={<GetByDateRange />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App
