import { apiRequest } from "./http";
import type { AuthRequest, AuthUser } from "./types";

export function register(body: AuthRequest): Promise<AuthUser> {
  return apiRequest<AuthUser>({ method: "POST", url: "/api/v1/auth/register", data: body });
}

export function login(body: AuthRequest): Promise<AuthUser> {
  return apiRequest<AuthUser>({ method: "POST", url: "/api/v1/auth/login", data: body });
}

export function logout(): Promise<void> {
  return apiRequest<void>({ method: "POST", url: "/api/v1/auth/logout" });
}

export function getCurrentUser(): Promise<AuthUser> {
  return apiRequest<AuthUser>({ method: "GET", url: "/api/v1/auth/me" });
}
