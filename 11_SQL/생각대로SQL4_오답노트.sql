-- 문제2번
select * from film limit 10;

select 
	rating,
	count(film_id)
from
	film f
where 
	rating in ('R', 'G')
group by
	rating;
	

-- 문제 4
  -- 정답 1
select * from actor limit 10;
select * from film_actor limit 10;

select 
	a.actor_id,
	a.first_name || ' ' || a.last_name as full_name,
	count(distinct film_id)
from
	actor a
	left outer join film_actor fa on a.actor_id = fa.actor_id
group by 
	a.actor_id
order by 
	a.actor_id;
	
  -- 정답 2
select
	d.*,
	a.first_name,
	a.last_name
from
	(
		select
			actor_id,
			count(distinct film_id) cnt
		from
			film_actor fa
		group by
			actor_id
	) as d
left outer join actor as a on d.actor_id = a.actor_id ;
		







