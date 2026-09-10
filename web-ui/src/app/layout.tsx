import type { Metadata, Viewport } from "next";
import "./globals.css";

/**
 * Phase 4 (W4/F10): PWA installability + iOS mobile polish.
 * - manifest.webmanifest with maskable icons (add-to-home-screen flow)
 * - apple-mobile-web-app meta: standalone display, status-bar styling
 * - theme-color that follows the light/dark toggle
 * - no-FOUC theme script: applies the persisted (or system) theme class
 *   before first paint — no white flash for dark-mode users.
 */
export const metadata: Metadata = {
  title: "LakeWind AI — Dongo-Dervio Wind Forecasts",
  description:
    "Hyperlocal, MOS bias-corrected wind forecasts for the Dongo-Dervio sailing corridor, Lake Como.",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "LakeWind AI",
  },
  icons: {
    icon: [
      { url: "/icons/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/icons/icon-192.png", sizes: "192x192" }],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f1f5f9" },
    { media: "(prefers-color-scheme: dark)", color: "#0b1220" },
  ],
};

const THEME_INIT = `(function(){try{var t=localStorage.getItem('lw_theme');var d=t==='dark'||(t===null&&window.matchMedia('(prefers-color-scheme: dark)').matches);if(d)document.documentElement.classList.add('dark');}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT }} />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
