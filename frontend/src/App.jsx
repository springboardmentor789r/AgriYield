import { Routes, Route} from "react-router-dom";
import HomePage from "./pages/HomePage";
import PredictPage from "./pages/PredictPage";
import ForecastPage from "./pages/ForecastPage";
import './App.css'

function App() {
  return (
    <Routes>
      <Route path = "/" element = {<HomePage />}/>
      <Route path = "/predict" element = {<PredictPage />}/>
      <Route path = "/forecast" element = {<ForecastPage/>} />
    </Routes>
  );
}

export default App
