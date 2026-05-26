/**
 * Class-name merger. `clsx` filters falsy values + flattens nested arrays;
 * `tailwind-merge` resolves Tailwind class conflicts (so `cn("p-2", "p-4")`
 * yields `p-4`). Used by every lifted component.
 */
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function confidenceClass(c: "HIGH" | "MODERATE" | "LOW" | null | undefined) {
  if (c === "HIGH") return "bg-confidence-high text-white";
  if (c === "MODERATE") return "bg-confidence-moderate text-black";
  if (c === "LOW") return "bg-confidence-low text-white";
  return "bg-neutral-300 text-black";
}
