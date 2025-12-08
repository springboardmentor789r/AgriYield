import { useNavigate } from "react-router-dom";
import Header from "../Components/Header";
import Footer from "../Components/Footer";
import homeBg from "../assets/home-bg-img.png";
import image1 from "../assets/image.png";
import image2 from "../assets/img2.png";

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="h-screen w-screen bg-cover bg-center flex flex-col" style={{ backgroundImage: `url(${homeBg})` }}>
            <Header />

            <div className="flex-1 flex flex-col justify-center items-center p-6 md:p-10">
                <h1 className="text-2xl md:text-4xl font-bold text-white text-center mb-6">
                    Welcome to the Dashboard
                </h1>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10 w-full max-w-6xl">
                    {/* Card 1 */}
                    <div className="bg-white p-4 md:p-6 rounded-xl shadow-lg flex flex-col items-center">
                        <img src={image1} alt="Tab 1" className="rounded-lg mb-4 w-full h-48 object-cover" />
                        <h2 className="text-lg md:text-xl font-bold mb-2 text-center">Predict CropYield by Parameters</h2>
                        <p className="text-gray-600 text-center mb-4">
                            Fill the required parameters to predict CropYield.
                        </p>
                        <button
                            onClick={() => navigate("/params")}
                            className="px-4 py-2 bg-green-600 text-white rounded-lg shadow-md"
                        >
                            Go to Parameters Form
                        </button>
                    </div>

                    {/* Card 2 */}
                    <div className="bg-white p-4 md:p-6 rounded-xl shadow-lg flex flex-col items-center">
                        <img src={image2} alt="Tab 2" className="rounded-lg mb-4 w-full h-48 object-cover" />
                        <h2 className="text-lg md:text-xl font-bold mb-2 text-center">Analysis of Predicted CropYield by parameters and Time Range</h2>
                        <p className="text-gray-600 text-center mb-4">
                            Enter start and end dates to predict the CropYield.
                        </p>
                        <button
                            onClick={() => navigate("/daterange")}
                            className="px-4 py-2 bg-green-600 text-white rounded-lg shadow-md"
                        >
                            Go to Paramters and Date Range Form
                        </button>
                    </div>
                </div>
            </div>

            <Footer />
        </div>

    );
};

export default Home;
