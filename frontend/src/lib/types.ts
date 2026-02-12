export interface SeasonMeta {
  id: string;
  display_name: string;
  year: number;
  type: 'regular' | 'vip';
  generated_at: string;
  latest_episode: number;
  total_episodes: number;
}

export interface SeasonIndexEntry {
  id: string;
  display_name: string;
  year: number;
  type: 'regular' | 'vip';
  current: boolean;
  latest_episode: number;
  total_episodes: number;
  num_males: number;
  num_females: number;
  total_solutions: number;
  confirmed_matches: number;
  infeasible: boolean;
  data_url: string;
}

export interface SeasonsIndex {
  generated_at: string;
  seasons: SeasonIndexEntry[];
}

export interface Contestants {
  males: string[];
  females: string[];
  is_unbalanced: boolean;
}

export interface SolverResult {
  total_solutions: number;
  solutions_capped: boolean;
  solved: boolean;
  infeasible: boolean;
}

export interface Pairing {
  male: string;
  female: string;
  probability: number;
  confirmed: boolean;
}

export interface ConfirmedMatch {
  male: string;
  female: string;
}

export interface DoubleMatchCandidate {
  name: string;
  probability: number;
  gender: 'male' | 'female';
}

export interface DoubleMatch {
  applicable: boolean;
  candidates: DoubleMatchCandidate[];
}

export interface MatchingNight {
  night_number: number;
  pairs: [string, string][];
  matches: number;
}

export interface SeasonData {
  meta: SeasonMeta;
  contestants: Contestants;
  solver_result: SolverResult;
  confirmed_matches: ConfirmedMatch[];
  ruled_out: ConfirmedMatch[];
  pairings: Pairing[];
  double_match: DoubleMatch;
  matching_nights: MatchingNight[];
}
