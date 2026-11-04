import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono, Noto_Sans_Devanagari } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const deva = Noto_Sans_Devanagari({
  variable: "--font-deva",
  subsets: ["devanagari", "latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Pipeline Evaluation",
  description: "Inspect extracted sentences and evaluate metadata classifiers",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${deva.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col font-sans">
        <header className="border-b border-border bg-surface">
          <div className="mx-auto max-w-6xl px-4 h-12 flex items-center gap-6">
            <Link href="/" className="font-semibold tracking-tight">
              Data Collection Pipeline
            </Link>
            <span className="text-sm text-muted">extract → sentences → classify → evaluate</span>
          </div>
        </header>
        <main className="flex-1 mx-auto w-full max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
