import { Link } from "react-router-dom";
import logo from "../assets/Logo.png";

const Header = () => {
  return (
    <header className="w-full shadow-xl font-poppins">
      <div className="flex justify-between items-center px-8 py-4
        bg-gradient-to-r from-green-400 via-green-500 to-yellow-500">

        {/* Logo + Title */}
        <div className="flex items-center gap-3">
          <img src={logo} alt="Logo" className="w-12 h-12 rounded-full shadow-md" />
          <h1 className="text-3xl font-extrabold text-white tracking-wide drop-shadow-lg">
            AgriYield Predictor
          </h1>
        </div>

        {/* Navigation 
        <nav className="flex items-center gap-6 text-white font-semibold text-lg">
          <Link to="/" className="hover:scale-110 transition">Home</Link>
          <Link to="/about" className="hover:scale-110 transition">About</Link>
          <Link to="/contact" className="hover:scale-110 transition">Contact</Link>
        </nav>*/}
      </div>
    </header>
  );
};

export default Header;
