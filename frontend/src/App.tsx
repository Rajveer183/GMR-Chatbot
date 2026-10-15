import React from "react";
import { FilterRequest } from "./api/client";
import { AIWorkspace } from "./components/AIWorkspace";

import "./styles/globals.css";

const defaultFilters: FilterRequest = {
  business_group: "All",
  business_sector_name: "All",
  unit_name: "All",
  customer_name: "All",
  project_name: "All",
  cpc: "All",
  ccn: "All"
};

export const App: React.FC = () => {
  return <AIWorkspace filters={defaultFilters} />;
};

export default App;
