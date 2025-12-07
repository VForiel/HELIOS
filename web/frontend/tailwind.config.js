/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                helios: {
                    dark: '#0f172a',
                    primary: '#3b82f6',
                    accent: '#8b5cf6',
                    surface: '#1e293b'
                }
            }
        },
    },
    plugins: [],
}
