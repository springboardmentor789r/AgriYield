/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        poppins: ["Poppins", "sans-serif"],
      },
      colors: {
        agri: {
          dark: "#14532d",
          medium: "#16a34a",
          light: "#bbf7d0",
          bg: "#f1fdf6",
        },
      },
      boxShadow: {
        soft: "0 18px 40px rgba(15, 118, 110, 0.14)",
      },
    },
  },
  plugins: [],
};
