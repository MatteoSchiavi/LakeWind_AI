import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LakeWind AI — Dongo-Dervio Wind Forecasts",
  description:
    "Hyperlocal, MOS bias-corrected wind forecasts for the Dongo-Dervio sailing corridor, Lake Como.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
