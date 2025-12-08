const Footer = () => {
  return (
    <footer className="text-white py-4 text-center text-xl font-bold shadow-md main-bg" >
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">

        {/* Left */}
        <div>
          <h2 className="text-xl font-bold">AgriYield Predictor</h2>
          <p className="text-sm mt-2">
            Smart agricultural insights for better decision-making.
          </p>
          <p className="text-sm mt-2">📩 support@agriyields.com</p>
          <p className="text-sm">📞 +91 xxxxx xxxxx</p>
        </div>

        {/* Center */}
        <div className="text-center">
          <h3 className="text-lg font-semibold mb-2">Quick Links</h3>
          <ul className="space-y-1 text-sm">
            <li><a href="/" className="hover:underline">Home</a></li>
            <li><a href="/params" className="hover:underline">Parameters Form to predict Crop Yield</a></li>
            <li><a href="/daterange" className="hover:underline">Time series analysis to predict Crop Yield over a Date Range</a></li>
            <li><a href="https://indiaai.gov.in/ministries/ministry-of-agriculture" className="hover:underline">About Us</a></li>
          </ul>
        </div>

        {/* Right */}
        <div className="text-right">
          <h3 className="text-lg font-semibold mb-2">Location</h3>
          <p className="text-sm">Hyderabad, Telangana, India</p>
          <p className="text-sm mt-4">
            © 2025 AgriYield Predictor <br /> All rights reserved
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
