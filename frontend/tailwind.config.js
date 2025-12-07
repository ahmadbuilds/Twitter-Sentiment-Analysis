/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#6366F1",   // Indigo
        secondary: "#F43F5E", // Pink/Red
        accent: "#10B981",    // Green
        background: "#F9FAFB",
        card: "#FFFFFF",
      },
      transitionProperty: {
        height: "height",
        spacing: "margin, padding",
        colors: "background-color, color, border-color",
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out forwards",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: 0, transform: "translateY(10px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      },
      borderRadius: {
        "xl": "1rem",
        "2xl": "1.5rem",
      },
    },
  },
  plugins: [],
};
