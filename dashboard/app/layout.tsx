import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import { requireSessionAuth } from "@/lib/auth";
import { GlobalAskClaude } from "@/components/GlobalAskClaude";
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

// Nav = the 6 consolidated surfaces (3 stores + 2 lenses + 1 external), in order.
const NAV_ITEMS: { href: string; label: string }[] = [
  { href: "/", label: "Overview" },
  { href: "/updates", label: "Updates" },
  { href: "/tasks", label: "Tasks" },
  { href: "/results", label: "Results" },
  { href: "/docs", label: "Docs" },
  { href: "/literature", label: "Literature" },
];

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Computed server-side so the global Ask-Claude panel only mounts for an
  // authenticated request. requireSessionAuth reads + verifies the session
  // cookie; when auth is disabled (DASHBOARD_AUTH_ENABLED!=true) it returns
  // null on a missing cookie, so the panel stays off until the dev-stub path
  // signs in — matching the public-by-default posture of /, /results.
  const authed = (await requireSessionAuth()) !== null;

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
              {NAV_ITEMS.map((item) => (
                <Link key={item.href} href={item.href} className="hover:text-stone-600">
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-10">
          {children}
        </main>
        <footer className="border-t border-stone-200 py-4 text-center text-xs text-stone-500">
          eps.superkaiba.com · single-user research dashboard
        </footer>
        <GlobalAskClaude authed={authed} />
      </body>
    </html>
  );
}
