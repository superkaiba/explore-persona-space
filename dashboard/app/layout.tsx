import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "EPS Dashboard",
  description: "Explore Persona Space — research task dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-stone-50 text-stone-900 flex flex-col">
        <header className="sticky top-0 z-30 border-b border-stone-200 bg-white/80 backdrop-blur">
          <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              EPS
            </Link>
            <nav className="flex items-center gap-4 text-sm">
              <Link href="/" className="hover:text-stone-600">
                Tasks
              </Link>
              <Link href="/updates" className="hover:text-stone-600">
                Updates
              </Link>
              <Link href="/log" className="hover:text-stone-600">
                Log
              </Link>
              <Link href="/literature" className="hover:text-stone-600">
                Literature
              </Link>
              <Link href="/docs" className="hover:text-stone-600">
                Docs
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-10">
          {children}
        </main>
        <footer className="border-t border-stone-200 py-4 text-center text-xs text-stone-500">
          eps.superkaiba.com · single-user research dashboard
        </footer>
      </body>
    </html>
  );
}
