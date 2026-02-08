-- Daily total demand with weather
SELECT sr.date,
       SUM(sr.requests) AS total_requests,
       w.temp_f,
       w.precip_in
FROM service_requests sr
JOIN weather_daily w ON sr.date = w.date
GROUP BY sr.date;

-- Event impact on demand
SELECT e.city_event,
       AVG(sr.requests) AS avg_requests
FROM service_requests sr
JOIN city_events e ON sr.date = e.date
GROUP BY e.city_event;

-- Category mix by zone
SELECT zone,
       category,
       AVG(requests) AS avg_requests
FROM service_requests
GROUP BY zone, category
ORDER BY zone, category;
